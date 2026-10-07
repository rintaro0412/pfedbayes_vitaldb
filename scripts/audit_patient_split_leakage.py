#!/usr/bin/env python3
"""Audit patient overlap across federated client and data-split boundaries.

The audit reads case placement from per-case NPZ filenames.  When a split has
no NPZ files, it falls back to ``contiguous/manifest.json`` and reads the
``case_segments`` entries.  Waveform arrays are never opened.

Only aggregate counts are emitted; subject identifiers are never included in
the JSON report or error messages.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


SPLITS: Tuple[str, ...] = ("train", "val", "test")
CASE_FILE_RE = re.compile(r"^case_(\d+)\.npz$")
MISSING_SUBJECT_TOKENS = frozenset({"", "na", "n/a", "nan", "none", "null"})


class AuditError(RuntimeError):
    """Raised when audit input is missing, ambiguous, or malformed."""


@dataclass(frozen=True)
class Placement:
    caseid: int
    client: str
    split: str


def _parse_caseid(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise AuditError(f"invalid caseid in {context}")
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        raise AuditError(f"invalid caseid in {context}")
    return int(text)


def _caseids_from_npz(split_dir: Path) -> List[int]:
    caseids: List[int] = []
    for path in sorted(split_dir.glob("case_*.npz")):
        if not path.is_file():
            continue
        match = CASE_FILE_RE.fullmatch(path.name)
        if match is None:
            raise AuditError(f"invalid case filename in {split_dir}: {path.name}")
        caseids.append(int(match.group(1)))
    return caseids


def _caseids_from_manifest(manifest_path: Path) -> List[int]:
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read contiguous manifest: {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise AuditError(f"contiguous manifest must contain a JSON object: {manifest_path}")
    segments = manifest.get("case_segments")
    if not isinstance(segments, list):
        raise AuditError(f"case_segments must be a list: {manifest_path}")

    # A large case may have more than one segment when it crosses shard
    # boundaries.  Such segments still represent one placement of one case.
    unique_caseids = set()
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or "caseid" not in segment:
            raise AuditError(f"invalid case_segments entry {index}: {manifest_path}")
        unique_caseids.add(
            _parse_caseid(segment["caseid"], context=f"{manifest_path} case_segments[{index}]")
        )

    total_cases = manifest.get("total_cases")
    if total_cases is not None:
        parsed_total = _parse_caseid(total_cases, context=f"{manifest_path} total_cases")
        if parsed_total != len(unique_caseids):
            raise AuditError(
                f"total_cases does not match unique case_segments caseids: {manifest_path}"
            )
    return sorted(unique_caseids)


def _enumerate_placements(data_dir: Path) -> Tuple[List[Placement], Dict[str, object]]:
    if not data_dir.is_dir():
        raise AuditError(f"data directory does not exist or is not a directory: {data_dir}")

    placements: List[Placement] = []
    split_dirs_by_source = {"npz": 0, "contiguous_manifest": 0}
    cases_by_source = {"npz": 0, "contiguous_manifest": 0}
    found_split_dir = False

    for client_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for split in SPLITS:
            split_dir = client_dir / split
            if not split_dir.exists():
                continue
            if not split_dir.is_dir():
                raise AuditError(f"split path is not a directory: {split_dir}")
            found_split_dir = True

            caseids = _caseids_from_npz(split_dir)
            if caseids:
                source = "npz"
            else:
                manifest_path = split_dir / "contiguous" / "manifest.json"
                if not manifest_path.is_file():
                    raise AuditError(
                        f"no case_*.npz or contiguous/manifest.json found in: {split_dir}"
                    )
                caseids = _caseids_from_manifest(manifest_path)
                source = "contiguous_manifest"

            split_dirs_by_source[source] += 1
            cases_by_source[source] += len(caseids)
            placements.extend(
                Placement(caseid=caseid, client=client_dir.name, split=split)
                for caseid in caseids
            )

    if not found_split_dir:
        raise AuditError(f"no <client>/<train|val|test> directories found in: {data_dir}")
    if not placements:
        raise AuditError(f"no cases found in federated data directory: {data_dir}")

    seen: Dict[int, Placement] = {}
    for placement in placements:
        previous = seen.get(placement.caseid)
        if previous is not None:
            raise AuditError(
                "duplicate case placement detected for "
                f"caseid={placement.caseid}: "
                f"{previous.client}/{previous.split} and "
                f"{placement.client}/{placement.split}"
            )
        seen[placement.caseid] = placement

    cases_by_split = {
        split: sum(1 for placement in placements if placement.split == split)
        for split in SPLITS
    }
    inventory: Dict[str, object] = {
        "clients": len({placement.client for placement in placements}),
        "split_directories": sum(split_dirs_by_source.values()),
        "split_directories_by_source": split_dirs_by_source,
        "cases_by_source": cases_by_source,
        "cases_by_split": cases_by_split,
    }
    return placements, inventory


def _load_subject_map(clinical_csv: Path) -> Dict[int, Optional[str]]:
    if not clinical_csv.is_file():
        raise AuditError(f"clinical CSV does not exist or is not a file: {clinical_csv}")

    subject_by_case: Dict[int, Optional[str]] = {}
    try:
        with clinical_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            required = {"caseid", "subjectid"}
            missing_columns = sorted(required - fieldnames)
            if missing_columns:
                raise AuditError(
                    "clinical CSV is missing required columns: " + ", ".join(missing_columns)
                )

            for row_number, row in enumerate(reader, start=2):
                caseid = _parse_caseid(
                    row.get("caseid", ""), context=f"{clinical_csv} row {row_number}"
                )
                if caseid in subject_by_case:
                    raise AuditError(
                        f"duplicate caseid in clinical CSV at row {row_number}: caseid={caseid}"
                    )
                raw_subject = str(row.get("subjectid") or "").strip()
                subject = (
                    None
                    if raw_subject.casefold() in MISSING_SUBJECT_TOKENS
                    else raw_subject
                )
                subject_by_case[caseid] = subject
    except (OSError, csv.Error) as exc:
        raise AuditError(f"cannot read clinical CSV: {clinical_csv}: {exc}") from exc

    return subject_by_case


def _metric_counts(
    placements_by_subject: Mapping[str, Sequence[Placement]],
    predicate,
) -> Tuple[int, int]:
    selected = [items for items in placements_by_subject.values() if predicate(items)]
    return len(selected), sum(len(items) for items in selected)


def _build_report(
    *,
    data_dir: Path,
    clinical_csv: Path,
    placements: Sequence[Placement],
    inventory: Mapping[str, object],
    subject_by_case: Mapping[int, Optional[str]],
) -> Dict[str, object]:
    placements_by_subject: Dict[str, List[Placement]] = defaultdict(list)
    missing_subject_cases = 0
    for placement in placements:
        subject = subject_by_case.get(placement.caseid)
        if subject is None:
            missing_subject_cases += 1
            continue
        placements_by_subject[subject].append(placement)

    repeat_subjects, repeat_cases = _metric_counts(
        placements_by_subject, lambda items: len(items) > 1
    )
    cross_client_subjects, cross_client_cases = _metric_counts(
        placements_by_subject,
        lambda items: len({placement.client for placement in items}) > 1,
    )
    cross_split_subjects, cross_split_cases = _metric_counts(
        placements_by_subject,
        lambda items: len({placement.split for placement in items}) > 1,
    )
    train_test_subjects, train_test_cases = _metric_counts(
        placements_by_subject,
        lambda items: {"train", "test"}.issubset(
            {placement.split for placement in items}
        ),
    )

    counts = {
        "total_cases": len(placements),
        "subjects_with_known_id": len(placements_by_subject),
        "missing_subject_cases": missing_subject_cases,
        "repeat_subjects": repeat_subjects,
        "repeat_cases": repeat_cases,
        "cross_client_subjects": cross_client_subjects,
        "cross_client_cases": cross_client_cases,
        "cross_split_subjects": cross_split_subjects,
        "cross_split_cases": cross_split_cases,
        "train_test_subjects": train_test_subjects,
        "train_test_cases": train_test_cases,
    }
    return {
        "schema_version": 1,
        "inputs": {
            "data_dir": str(data_dir),
            "clinical_csv": str(clinical_csv),
        },
        "inventory": dict(inventory),
        "counts": counts,
    }


def _write_report(report: Mapping[str, object], output: Optional[Path]) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None or str(output) == "-":
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit repeated-patient overlap across federated clients and "
            "train/val/test splits without reading waveform arrays."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Federated dataset root containing <client>/<split>/ directories.",
    )
    parser.add_argument(
        "--clinical-csv",
        required=True,
        type=Path,
        help="Clinical CSV containing caseid and subjectid columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; omit or use '-' for stdout.",
    )
    parser.add_argument(
        "--fail-on-cross-split",
        action="store_true",
        help="Exit with status 2 after reporting when any subject crosses splits.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        placements, inventory = _enumerate_placements(args.data_dir)
        subject_by_case = _load_subject_map(args.clinical_csv)
        report = _build_report(
            data_dir=args.data_dir,
            clinical_csv=args.clinical_csv,
            placements=placements,
            inventory=inventory,
            subject_by_case=subject_by_case,
        )
        _write_report(report, args.output)
    except (AuditError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.fail_on_cross_split and report["counts"]["cross_split_subjects"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

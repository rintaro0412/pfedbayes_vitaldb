from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_git_hash(repo_root: str | Path | None = None) -> str | None:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def read_caseids(
    *,
    caseids: str | None = None,
    caseids_file: str | Path | None = None,
    caseid_column: str = "caseid",
) -> list[int]:
    out: list[int] = []

    if caseids:
        for tok in str(caseids).split(","):
            tok = tok.strip()
            if not tok:
                continue
            out.append(int(tok))

    if caseids_file is not None:
        p = Path(caseids_file)
        if not p.exists():
            raise FileNotFoundError(f"caseids_file not found: {p}")

        if p.suffix.lower() in {".csv"}:
            with p.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                if caseid_column not in (r.fieldnames or []):
                    raise ValueError(f"CSV missing column '{caseid_column}': {p}")
                for row in r:
                    v = row.get(caseid_column)
                    if v is None:
                        continue
                    v = str(v).strip()
                    if v:
                        out.append(int(v))
        else:
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s:
                    continue
                out.append(int(s))

    # de-dup preserving order
    seen: set[int] = set()
    uniq: list[int] = []
    for cid in out:
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(int(cid))
    return uniq


@dataclass(frozen=True)
class ClinicalMeta:
    caseid: int
    subjectid: int | None
    department: str | None


def load_clinical_csv(path: str | Path):
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"clinical csv not found: {p}")
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    if "caseid" not in df.columns:
        raise ValueError(f"clinical csv missing 'caseid': {p}")
    return df


def build_clinical_lookup(df) -> dict[int, ClinicalMeta]:
    cols = {c: c for c in df.columns}
    has_subject = "subjectid" in cols
    has_dept = "department" in cols
    out: dict[int, ClinicalMeta] = {}
    for _, r in df.iterrows():
        try:
            caseid = int(r["caseid"])
        except Exception:
            continue
        subjectid = None
        if has_subject:
            try:
                subjectid = int(r["subjectid"])
            except Exception:
                subjectid = None
        dept = None
        if has_dept:
            d = r.get("department")
            if d is not None and str(d).strip() != "":
                dept = str(d).strip()
        out[caseid] = ClinicalMeta(caseid=caseid, subjectid=subjectid, department=dept)
    return out


def resolve_project_root(from_file: str | Path) -> Path:
    """Return repo root assuming this file lives under it (1-2 levels deep)."""
    return Path(from_file).resolve().parent.parent


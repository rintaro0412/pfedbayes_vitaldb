from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure project root in sys.path when executed as `python scripts/make_server_unlabeled.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import list_npz_files, parse_caseid_from_path


def _collect_train_files(federated_dir: str) -> List[str]:
    return list_npz_files(federated_dir, "train")


def _collect_split_caseids(federated_dir: str, split: str) -> List[int]:
    files = list_npz_files(federated_dir, split)
    out: List[int] = []
    for p in files:
        cid = parse_caseid_from_path(p)
        if cid is None:
            continue
        out.append(int(cid))
    return out


def _summarize_exclusions(federated_dir: str) -> Dict[str, int]:
    val_files = list_npz_files(federated_dir, "val")
    test_files = list_npz_files(federated_dir, "test")
    return {
        "val_files": int(len(val_files)),
        "test_files": int(len(test_files)),
    }


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _save_npz(
    out_path: str,
    *,
    x_wave: np.ndarray,
    x_clin: np.ndarray | None,
    y: np.ndarray | None,
) -> None:
    if x_clin is None:
        if y is None:
            np.savez_compressed(out_path, x_wave=x_wave)
        else:
            np.savez_compressed(out_path, x_wave=x_wave, y=y)
        return
    if y is None:
        np.savez_compressed(out_path, x_wave=x_wave, x_clin=x_clin)
        return
    np.savez_compressed(out_path, x_wave=x_wave, x_clin=x_clin, y=y)


def _extract_client_id(path: str) -> str | None:
    p = Path(path)
    if p.parent.name != "train":
        return None
    return p.parent.parent.name if p.parent.parent is not None else None


def create_server_unlabeled(
    *,
    federated_data_dir: str,
    out_dir: str,
    frac: float,
    seed: int,
    mode: str,
) -> Dict[str, int]:
    rng = np.random.default_rng(int(seed))
    train_files = _collect_train_files(federated_data_dir)
    exclusions = _summarize_exclusions(federated_data_dir)

    unique: Dict[int, str] = {}
    source_client: Dict[int, str] = {}
    dup_count = 0
    missing_caseid = 0

    for p in train_files:
        caseid = parse_caseid_from_path(p)
        if caseid is None:
            missing_caseid += 1
            continue
        if caseid in unique:
            dup_count += 1
            continue
        unique[int(caseid)] = p
        cid = _extract_client_id(p)
        if cid is not None:
            source_client[int(caseid)] = str(cid)

    caseids = sorted(unique.keys())
    total_unique = int(len(caseids))
    if frac <= 0:
        n_select = 0
    else:
        n_select = int(round(total_unique * float(frac)))
        n_select = max(1, min(n_select, total_unique))

    chosen = set()
    if n_select > 0:
        chosen = set(rng.choice(caseids, size=int(n_select), replace=False).tolist())

    os.makedirs(out_dir, exist_ok=True)

    written = 0
    selected_by_client: Dict[str, int] = {}
    for cid in sorted(chosen):
        src = unique[int(cid)]
        src_client = source_client.get(int(cid))
        data = _load_npz(src)
        if "x_wave" not in data:
            continue
        x_wave = data["x_wave"]
        x_clin = data.get("x_clin")
        if mode == "drop_y":
            y = None
        elif mode == "dummy_y":
            y = data.get("y")
            if y is None:
                y = np.zeros((int(x_wave.shape[0]),), dtype=np.int64)
            else:
                y = np.zeros_like(np.asarray(y), dtype=np.int64)
        else:
            raise ValueError(f"unknown mode: {mode}")

        dst = os.path.join(out_dir, f"case_{int(cid)}.npz")
        _save_npz(dst, x_wave=x_wave, x_clin=x_clin, y=y)
        written += 1
        if src_client is not None:
            selected_by_client[src_client] = selected_by_client.get(src_client, 0) + 1

    # leakage check: server_unlabeled vs val/test
    val_caseids = set(_collect_split_caseids(federated_data_dir, "val"))
    test_caseids = set(_collect_split_caseids(federated_data_dir, "test"))
    unlabeled_caseids = set(int(c) for c in chosen)
    overlap_val = sorted(list(unlabeled_caseids & val_caseids))
    overlap_test = sorted(list(unlabeled_caseids & test_caseids))

    overlap_report = {
        "overlap_val_count": int(len(overlap_val)),
        "overlap_test_count": int(len(overlap_test)),
        "overlap_val_sample": overlap_val[:10],
        "overlap_test_sample": overlap_test[:10],
        "selected_by_client": {k: int(v) for k, v in sorted(selected_by_client.items())},
    }
    report_path = Path(out_dir) / "overlap_report.json"
    report_path.write_text(json.dumps(overlap_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "train_files": int(len(train_files)),
        "train_unique_cases": int(total_unique),
        "selected_cases": int(len(chosen)),
        "written": int(written),
        "excluded_val_files": int(exclusions["val_files"]),
        "excluded_test_files": int(exclusions["test_files"]),
        "excluded_duplicates": int(dup_count),
        "excluded_missing_caseid": int(missing_caseid),
        "overlap_val_count": int(len(overlap_val)),
        "overlap_test_count": int(len(overlap_test)),
        "overlap_report": str(report_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Make server-side unlabeled dataset from federated train split")
    ap.add_argument("--federated-data-dir", default="federated_data")
    ap.add_argument("--out-dir", default="server_unlabeled")
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="drop_y", choices=["drop_y", "dummy_y"])
    args = ap.parse_args()

    stats = create_server_unlabeled(
        federated_data_dir=str(args.federated_data_dir),
        out_dir=str(args.out_dir),
        frac=float(args.frac),
        seed=int(args.seed),
        mode=str(args.mode),
    )

    print("[server_unlabeled]")
    print(f"  train_files: {stats['train_files']}")
    print(f"  train_unique_cases: {stats['train_unique_cases']}")
    print(f"  selected_cases: {stats['selected_cases']}")
    print(f"  written: {stats['written']}")
    print(f"  excluded_val_files: {stats['excluded_val_files']}")
    print(f"  excluded_test_files: {stats['excluded_test_files']}")
    print(f"  excluded_duplicates: {stats['excluded_duplicates']}")
    print(f"  excluded_missing_caseid: {stats['excluded_missing_caseid']}")
    print(f"  overlap_val_count: {stats['overlap_val_count']}")
    print(f"  overlap_test_count: {stats['overlap_test_count']}")
    print(f"  overlap_report: {stats['overlap_report']}")


if __name__ == "__main__":
    main()

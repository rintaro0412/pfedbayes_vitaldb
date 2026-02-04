from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Ensure project root in sys.path when executed as `python scripts/make_table1_client_summary.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io import now_utc_iso, write_json


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scan_counts(data_dir: Path, splits: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    clients = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    stats: Dict[str, Dict[str, Dict[str, float]]] = {c: {s: {"n_files": 0.0, "n_windows": 0.0, "n_pos": 0.0, "n_neg": 0.0} for s in splits} for c in clients}
    for c in clients:
        for s in splits:
            files = sorted((data_dir / c / s).glob("*.npz"))
            stats[c][s]["n_files"] = float(len(files))
            for f in files:
                with np.load(f, allow_pickle=False) as z:
                    if "y" not in z:
                        continue
                    y = np.asarray(z["y"], dtype=np.int64)
                    n = int(y.size)
                    pos = int((y > 0).sum())
                    stats[c][s]["n_windows"] += float(n)
                    stats[c][s]["n_pos"] += float(pos)
                    stats[c][s]["n_neg"] += float(n - pos)
    return stats


def _pooled(stats: Dict[str, Dict[str, Dict[str, float]]], splits: List[str]) -> Dict[str, Dict[str, int]]:
    pooled: Dict[str, Dict[str, int]] = {}
    for c, per_split in stats.items():
        n_files = int(sum(per_split[s]["n_files"] for s in splits))
        n_windows = int(sum(per_split[s]["n_windows"] for s in splits))
        n_pos = int(sum(per_split[s]["n_pos"] for s in splits))
        n_neg = int(sum(per_split[s]["n_neg"] for s in splits))
        pooled[c] = {
            "n_files": n_files,
            "n_windows": n_windows,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }
    return pooled


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Table1 client summary from existing artifacts")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--noniid-json", default="tmp_noniid_report.json")
    ap.add_argument("--summary-json", default="federated_data/summary.json")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--split-unit", default="case", help="Split unit label to record in table (default=case)")
    ap.add_argument("--out-csv", default="outputs/table1_client_summary.csv")
    ap.add_argument("--out-json", default="outputs/table1_client_summary.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    noniid = _load_json(Path(args.noniid_json))
    summary = _load_json(Path(args.summary_json))

    if noniid and "per_client_per_split" in noniid:
        stats = noniid["per_client_per_split"]
    else:
        stats = _scan_counts(data_dir, splits)

    pooled = _pooled(stats, splits)

    exclusion_rate = None
    missing_rate = None
    missing_rates_by_col = None
    if summary:
        raw_cases = summary.get("exclusion", {}).get("raw_cases")
        eligible_cases = summary.get("eligible_cases")
        if raw_cases:
            try:
                exclusion_rate = 1.0 - (float(eligible_cases) / float(raw_cases))
            except Exception:
                exclusion_rate = None
        missing_counts = summary.get("missing", {}).get("counts", {})
        missing_rates_by_col = summary.get("missing", {}).get("rates")
        after_basic = summary.get("exclusion", {}).get("after_basic_filter")
        if missing_counts and after_basic:
            total_missing = float(sum(float(v) for v in missing_counts.values()))
            denom = float(after_basic) * float(len(missing_counts))
            missing_rate = (total_missing / denom) if denom > 0 else None

    rows = []
    for c in sorted(pooled.keys()):
        p = pooled[c]
        n_windows = int(p.get("n_windows", 0))
        n_pos = int(p.get("n_pos", 0))
        pos_rate = (float(n_pos) / float(n_windows)) if n_windows > 0 else float("nan")
        row = {
            "client_id": c,
            "n_samples": int(n_windows),
            "n_pos": int(n_pos),
            "pos_rate": float(pos_rate),
            "train_cases": int(stats[c]["train"]["n_files"]) if "train" in stats[c] else 0,
            "val_cases": int(stats[c]["val"]["n_files"]) if "val" in stats[c] else 0,
            "test_cases": int(stats[c]["test"]["n_files"]) if "test" in stats[c] else 0,
            "train_windows": int(stats[c]["train"]["n_windows"]) if "train" in stats[c] else 0,
            "val_windows": int(stats[c]["val"]["n_windows"]) if "val" in stats[c] else 0,
            "test_windows": int(stats[c]["test"]["n_windows"]) if "test" in stats[c] else 0,
            "exclusion_rate": exclusion_rate,
            "missing_rate": missing_rate,
            "split_unit": str(args.split_unit),
        }
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    payload = {
        "started_utc": now_utc_iso(),
        "data_dir": str(data_dir),
        "splits": splits,
        "split_unit": str(args.split_unit),
        "exclusion_rate": exclusion_rate,
        "missing_rate": missing_rate,
        "missing_rates_by_col": missing_rates_by_col,
        "rows": rows,
        "finished_utc": now_utc_iso(),
    }
    write_json(out_json, payload)
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()

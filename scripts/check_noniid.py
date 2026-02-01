from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root in sys.path when executed as `python scripts/check_noniid.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io import now_utc_iso, write_json


CLIN_COLS = [
    "age",
    "sex_M",
    "bmi",
    "asa",
    "emop",
    "preop_htn",
    "preop_hb",
    "preop_bun",
    "preop_cr",
    "preop_alb",
    "preop_na",
    "preop_k",
]


def _chi2_label_test(counts: np.ndarray) -> Dict[str, float]:
    """
    counts: (n_clients, 2) => [pos, neg]
    """
    o = np.asarray(counts, dtype=np.float64)
    if o.ndim != 2 or o.shape[1] != 2:
        raise ValueError(f"counts must be (n_clients,2), got {o.shape}")
    row = o.sum(axis=1, keepdims=True)
    col = o.sum(axis=0, keepdims=True)
    total = float(o.sum())
    if total <= 0:
        return {"chi2": float("nan"), "df": float("nan"), "p_value": float("nan")}
    e = row * col / total
    chi2 = float(((o - e) ** 2 / np.maximum(e, 1e-12)).sum())
    df = float(max(int(o.shape[0] - 1), 0))
    out = {"chi2": float(chi2), "df": float(df)}
    try:
        import scipy.stats as st  # type: ignore

        out["p_value"] = float(st.chi2.sf(chi2, int(df)))
    except Exception:
        out["p_value"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Check non-IID across federated clients (label + covariates)")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--splits", default="train,val,test", help="Comma-separated splits to include")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if not splits:
        raise SystemExit("No splits specified.")

    clients = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if not clients:
        raise SystemExit(f"No clients found under {data_dir}")

    # stats[client][split] accumulators
    stats: Dict[str, Dict[str, Dict[str, float]]] = {c: {s: defaultdict(float) for s in splits} for c in clients}
    case_pos_rates: Dict[str, List[float]] = defaultdict(list)
    case_n_windows: Dict[str, List[int]] = defaultdict(list)
    clin_by_client: Dict[str, List[np.ndarray]] = defaultdict(list)

    for c in clients:
        for s in splits:
            files = sorted((data_dir / c / s).glob("*.npz"))
            stats[c][s]["n_files"] = float(len(files))
            for f in files:
                with np.load(f, allow_pickle=False) as z:
                    y = np.asarray(z["y"], dtype=np.int64)
                    n = int(y.size)
                    pos = int((y > 0).sum())
                    stats[c][s]["n_windows"] += float(n)
                    stats[c][s]["n_pos"] += float(pos)
                    stats[c][s]["n_neg"] += float(n - pos)
                    if n > 0:
                        case_pos_rates[c].append(float(pos / n))
                        case_n_windows[c].append(int(n))
                    if "x_clin" in z:
                        xc = np.asarray(z["x_clin"])
                        if xc.ndim == 2 and xc.shape[0] > 0:
                            clin_by_client[c].append(np.asarray(xc[0], dtype=np.float64))

    # pooled stats per client
    pooled = {}
    for c in clients:
        n_files = int(sum(stats[c][s]["n_files"] for s in splits))
        n_windows = int(sum(stats[c][s]["n_windows"] for s in splits))
        n_pos = int(sum(stats[c][s]["n_pos"] for s in splits))
        n_neg = int(sum(stats[c][s]["n_neg"] for s in splits))
        pooled[c] = {
            "n_files": int(n_files),
            "n_windows": int(n_windows),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "pos_rate": float(n_pos / n_windows) if n_windows else float("nan"),
        }

    case_rate_summary = {}
    for c in clients:
        arr = np.asarray(case_pos_rates[c], dtype=np.float64)
        if arr.size == 0:
            case_rate_summary[c] = {"n_cases": 0}
            continue
        case_rate_summary[c] = {
            "n_cases": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    case_windows_summary = {}
    for c in clients:
        arr = np.asarray(case_n_windows[c], dtype=np.float64)
        if arr.size == 0:
            case_windows_summary[c] = {"n_cases": 0}
            continue
        case_windows_summary[c] = {
            "n_cases": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    clin_summary = {}
    for c in clients:
        x = np.asarray(clin_by_client[c], dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            clin_summary[c] = {"n_cases": int(x.shape[0]) if x.ndim == 2 else 0, "mean": None}
            continue
        mu = x.mean(axis=0)
        clin_summary[c] = {"n_cases": int(x.shape[0]), "mean": {k: float(v) for k, v in zip(CLIN_COLS, mu.tolist())}}

    counts = np.array([[pooled[c]["n_pos"], pooled[c]["n_neg"]] for c in clients], dtype=np.float64)
    chi2 = _chi2_label_test(counts)

    report: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "data_dir": str(data_dir),
        "splits": list(splits),
        "clients": list(clients),
        "per_client_per_split": stats,
        "per_client_pooled": pooled,
        "case_pos_rate": case_rate_summary,
        "case_n_windows": case_windows_summary,
        "clinical_mean": clin_summary,
        "label_chi2_test": chi2,
        "finished_utc": now_utc_iso(),
    }

    # stdout (compact)
    print("clients:", ", ".join(clients))
    print("label pos_rate (pooled):")
    for c in clients:
        p = pooled[c]["pos_rate"]
        print(f"  {c:16s} n_windows={pooled[c]['n_windows']:7d} pos_rate={p:.4f}")
    p_val = float(chi2.get("p_value", float("nan")))
    if np.isfinite(p_val):
        print(f"chi2(label~client): chi2={chi2['chi2']:.3f} df={int(chi2['df'])} p={p_val:.3e}")
    else:
        print(f"chi2(label~client): chi2={chi2['chi2']:.3f} df={int(chi2['df'])} (p_value unavailable)")

    out_path = args.out
    if out_path:
        write_json(out_path, report)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

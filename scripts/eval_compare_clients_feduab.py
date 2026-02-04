from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root in sys.path when executed as `python scripts/eval_compare_clients_feduab.py`
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.eval_summary import METRIC_KEYS, build_mode_report, write_eval_outputs
from common.io import now_utc_iso, read_json, write_json


def _load_summary_min_cases(data_dir: str, default: int) -> int:
    p = Path(data_dir) / "summary.json"
    if not p.exists():
        return int(default)
    try:
        js = read_json(p)
        return int(js.get("min_client_cases", default))
    except Exception:
        return int(default)


def _ensure_eval_modes_fedavg(run_dir: Path) -> None:
    out_path = run_dir / "eval_modes.json"
    if out_path.exists():
        return
    test_report = run_dir / "test_report.json"
    per_client_csv = run_dir / "test_report_per_client.csv"
    if not test_report.exists() or not per_client_csv.exists():
        raise FileNotFoundError("FedAvg outputs not found for eval_modes conversion.")

    rep = read_json(test_report)
    metrics_pre = rep.get("metrics_pre", {}) if isinstance(rep, dict) else {}
    overall = {k: metrics_pre.get(k, float("nan")) for k in ("n", "n_pos", "n_neg", *METRIC_KEYS)}

    def _sf(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    per_client: Dict[str, Dict[str, Any]] = {}
    with per_client_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") not in (None, "", "ok"):
                continue
            cid = str(row.get("client_id", ""))
            if not cid:
                continue
            per_client[cid] = {
                "n": _sf(row.get("n", float("nan"))),
                "n_pos": _sf(row.get("n_pos", float("nan"))),
                "n_neg": _sf(row.get("n_neg", float("nan"))),
                "auroc": _sf(row.get("auroc_pre", float("nan"))),
                "auprc": _sf(row.get("auprc_pre", float("nan"))),
                "ece": _sf(row.get("ece_pre", float("nan"))),
                "nll": _sf(row.get("nll_pre", float("nan"))),
                "brier": _sf(row.get("brier_pre", float("nan"))),
            }

    modes = {
        "global_idless": build_mode_report(overall=overall, per_client=per_client),
        "personalized_oracle": build_mode_report(overall=overall, per_client=per_client, note="FedAvg uses global model"),
        "ensemble_idless": build_mode_report(overall=overall, per_client=per_client, note="FedAvg uses global model"),
    }
    write_eval_outputs(run_dir=run_dir, algo="fedavg", modes=modes)


def _load_eval_modes(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "eval_modes.json"
    if not path.exists():
        raise FileNotFoundError(f"eval_modes.json not found: {path}")
    return read_json(path)


def _select_mode(payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    modes = payload.get("modes", {}) if isinstance(payload, dict) else {}
    if mode not in modes:
        raise KeyError(f"mode '{mode}' not found in eval_modes.json")
    return modes[mode]


def _macro_from_clients(per_client: Dict[str, Any], eligible: set[str]) -> Dict[str, float]:
    vals = {k: [] for k in METRIC_KEYS}
    for cid, rec in per_client.items():
        if cid not in eligible:
            continue
        for k in METRIC_KEYS:
            vals[k].append(float(rec.get(k, float("nan"))))
    out = {}
    for k, arr in vals.items():
        v = [x for x in arr if x == x]
        out[k] = float(np.mean(v)) if v else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare FedAvg vs FedUAB using eval_modes outputs.")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--fedavg-run", required=True)
    ap.add_argument("--feduab-run", required=True)
    ap.add_argument("--mode", default="global_idless", choices=["global_idless", "personalized_oracle", "ensemble_idless"])
    ap.add_argument("--min-client-cases", type=int, default=None)
    ap.add_argument("--seed-label", default="seed0")
    ap.add_argument("--out-json", default="compare_clients_feduab.json")
    ap.add_argument("--out-csv", default="compare_clients_feduab.csv")
    args = ap.parse_args()

    fedavg_run = Path(args.fedavg_run)
    feduab_run = Path(args.feduab_run)

    _ensure_eval_modes_fedavg(fedavg_run)
    fedavg_payload = _load_eval_modes(fedavg_run)
    feduab_payload = _load_eval_modes(feduab_run)

    fedavg_mode = _select_mode(fedavg_payload, args.mode)
    feduab_mode = _select_mode(feduab_payload, args.mode)

    per_client_fedavg = fedavg_mode.get("per_client", {}) or {}
    per_client_feduab = feduab_mode.get("per_client", {}) or {}

    min_cases = int(args.min_client_cases) if args.min_client_cases is not None else _load_summary_min_cases(args.data_dir, 150)
    eligible = {cid for cid, rec in per_client_fedavg.items() if float(rec.get("n", 0)) >= min_cases}
    excluded = {cid for cid in per_client_fedavg if cid not in eligible}

    macro_fedavg = _macro_from_clients(per_client_fedavg, eligible)
    macro_feduab = _macro_from_clients(per_client_feduab, eligible)
    diff_macro = {k: float(macro_feduab.get(k, float("nan")) - macro_fedavg.get(k, float("nan"))) for k in METRIC_KEYS}

    results = {
        "started_utc": now_utc_iso(),
        "data_dir": str(args.data_dir),
        "mode": str(args.mode),
        "min_client_cases": int(min_cases),
        "seed": str(args.seed_label),
        "fedavg_run": str(fedavg_run),
        "feduab_run": str(feduab_run),
        "eligible_clients": sorted(list(eligible)),
        "excluded_clients": sorted(list(excluded)),
        "macro": {"fedavg": macro_fedavg, "feduab": macro_feduab, "diff": diff_macro},
        "clients": {
            cid: {
                "fedavg": per_client_fedavg.get(cid, {}),
                "feduab": per_client_feduab.get(cid, {}),
                "eligible": cid in eligible,
            }
            for cid in sorted(per_client_fedavg.keys())
        },
        "finished_utc": now_utc_iso(),
    }

    write_json(args.out_json, results)

    # CSV (seed,method,metrics)
    rows = [
        {"seed": str(args.seed_label), "method": "fedavg", **macro_fedavg},
        {"seed": str(args.seed_label), "method": "feduab", **macro_feduab},
    ]
    import pandas as pd

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(json.dumps(results["macro"], indent=2, ensure_ascii=False))
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV:  {args.out_csv}")


if __name__ == "__main__":
    main()

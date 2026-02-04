from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover
    stats = None

# Ensure project root in sys.path when executed as `python scripts/ttest_feduab_vs_fedavg.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.eval_summary import METRIC_KEYS
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

    from common.eval_summary import build_mode_report, write_eval_outputs

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


def _macro_from_clients(per_client: Dict[str, Any], eligible: set[str], metrics: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in metrics:
        vals = []
        for cid, rec in per_client.items():
            if cid not in eligible:
                continue
            try:
                v = float(rec.get(k, float("nan")))
            except Exception:
                v = float("nan")
            if v == v:
                vals.append(v)
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def _paired_stats(
    fedavg_vals: List[float],
    feduab_vals: List[float],
    labels: List[str],
) -> Dict[str, Any]:
    pairs: List[Tuple[float, float, str]] = []
    for a, b, lab in zip(fedavg_vals, feduab_vals, labels):
        if a == a and b == b and math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b, lab))
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "fedavg_mean": float("nan"),
            "feduab_mean": float("nan"),
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "effect_d": float("nan"),
            "pairs": [],
        }
    a = np.asarray([p[0] for p in pairs], dtype=np.float64)
    b = np.asarray([p[1] for p in pairs], dtype=np.float64)
    diffs = b - a
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if n >= 2 else float("nan")
    t_stat = float("nan")
    p_value = float("nan")
    if n >= 2 and stats is not None:
        res = stats.ttest_rel(b, a, nan_policy="omit")
        try:
            t_stat = float(res.statistic)
            p_value = float(res.pvalue)
        except Exception:
            t_stat = float("nan")
            p_value = float("nan")
    effect_d = float(mean_diff / std_diff) if std_diff and std_diff == std_diff and std_diff != 0 else float("nan")
    return {
        "n": int(n),
        "fedavg_mean": float(a.mean()),
        "feduab_mean": float(b.mean()),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "effect_d": float(effect_d),
        "pairs": [{"label": lab, "fedavg": float(x), "feduab": float(y)} for x, y, lab in pairs],
    }


def _list_common_run_tags(fedavg_root: Path, feduab_root: Path) -> List[str]:
    if not fedavg_root.exists() or not feduab_root.exists():
        return []
    a = {p.name for p in fedavg_root.iterdir() if p.is_dir()}
    b = {p.name for p in feduab_root.iterdir() if p.is_dir()}
    return sorted(a & b)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired t-test: FedAvg vs FedUAB (overall + client-macro).")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--fedavg-root", default="runs/fedavg")
    ap.add_argument("--feduab-root", default="runs/feduab")
    ap.add_argument("--mode", default="global_idless", choices=["global_idless", "personalized_oracle", "ensemble_idless"])
    ap.add_argument("--run-tags", default="", help="Comma-separated run tags to include (e.g., seed0,seed1).")
    ap.add_argument("--metrics", default="auroc,auprc,nll,ece,brier")
    ap.add_argument("--min-client-cases", type=int, default=None)
    ap.add_argument("--out-json", default="outputs_uab/feduab_vs_fedavg_ttest.json")
    ap.add_argument("--out-csv", default="outputs_uab/feduab_vs_fedavg_ttest.csv")
    ap.add_argument("--min-seeds", type=int, default=2)
    args = ap.parse_args()

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified.")

    fedavg_root = Path(args.fedavg_root)
    feduab_root = Path(args.feduab_root)
    tags = _list_common_run_tags(fedavg_root, feduab_root)
    if args.run_tags.strip():
        keep = {t.strip() for t in str(args.run_tags).split(",") if t.strip()}
        tags = [t for t in tags if t in keep]

    if not tags:
        raise SystemExit("No common run tags found.")

    min_cases = int(args.min_client_cases) if args.min_client_cases is not None else _load_summary_min_cases(args.data_dir, 150)

    per_seed: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tag in tags:
        fedavg_run = fedavg_root / tag
        feduab_run = feduab_root / tag
        _ensure_eval_modes_fedavg(fedavg_run)
        fedavg_payload = _load_eval_modes(fedavg_run)
        feduab_payload = _load_eval_modes(feduab_run)

        fedavg_mode = _select_mode(fedavg_payload, args.mode)
        feduab_mode = _select_mode(feduab_payload, args.mode)

        overall_fedavg = fedavg_mode.get("overall", {}) or {}
        overall_feduab = feduab_mode.get("overall", {}) or {}

        per_client_fedavg = fedavg_mode.get("per_client", {}) or {}
        per_client_feduab = feduab_mode.get("per_client", {}) or {}

        eligible = {cid for cid, rec in per_client_fedavg.items() if float(rec.get("n", 0)) >= min_cases}

        macro_fedavg = _macro_from_clients(per_client_fedavg, eligible, metrics)
        macro_feduab = _macro_from_clients(per_client_feduab, eligible, metrics)

        per_seed[tag] = {
            "overall_fedavg": {k: float(overall_fedavg.get(k, float("nan"))) for k in metrics},
            "overall_feduab": {k: float(overall_feduab.get(k, float("nan"))) for k in metrics},
            "macro_fedavg": macro_fedavg,
            "macro_feduab": macro_feduab,
        }

    # Prepare paired arrays
    labels = list(per_seed.keys())
    results: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "data_dir": str(args.data_dir),
        "mode": str(args.mode),
        "min_client_cases": int(min_cases),
        "metrics": list(metrics),
        "run_tags": labels,
        "note": "paired t-test across seeds; if n<min_seeds, p_value is NaN",
        "overall": {},
        "macro": {},
        "per_seed": per_seed,
    }

    for scope in ("overall", "macro"):
        for k in metrics:
            a = [per_seed[tag][f"{scope}_fedavg"][k] for tag in labels]
            b = [per_seed[tag][f"{scope}_feduab"][k] for tag in labels]
            stats_out = _paired_stats(a, b, labels)
            stats_out["metric"] = k
            stats_out["scope"] = scope
            results[scope][k] = stats_out

    results["finished_utc"] = now_utc_iso()

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    write_json(str(out_json), results)

    # CSV summary
    rows = []
    for scope in ("overall", "macro"):
        for k in metrics:
            rec = results[scope][k]
            rows.append(
                {
                    "scope": scope,
                    "metric": k,
                    "n": rec.get("n", 0),
                    "fedavg_mean": rec.get("fedavg_mean", float("nan")),
                    "feduab_mean": rec.get("feduab_mean", float("nan")),
                    "mean_diff": rec.get("mean_diff", float("nan")),
                    "std_diff": rec.get("std_diff", float("nan")),
                    "t_stat": rec.get("t_stat", float("nan")),
                    "p_value": rec.get("p_value", float("nan")),
                    "effect_d": rec.get("effect_d", float("nan")),
                }
            )
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "metric",
                "n",
                "fedavg_mean",
                "feduab_mean",
                "mean_diff",
                "std_diff",
                "t_stat",
                "p_value",
                "effect_d",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    n_seeds = len(labels)
    if n_seeds < int(args.min_seeds):
        print(f"[WARN] seeds={n_seeds} < min_seeds={int(args.min_seeds)}; p_value is NaN")
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV:  {out_csv}")


if __name__ == "__main__":
    main()

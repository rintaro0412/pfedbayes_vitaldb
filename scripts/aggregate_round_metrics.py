from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_config(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def _load_round_client_metrics(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "round_client_metrics.csv"
    json_path = run_dir / "round_client_metrics.json"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        return pd.DataFrame(json.loads(json_path.read_text(encoding="utf-8")))
    per_round = sorted(run_dir.glob("round_*_test_per_client.csv"))
    if not per_round:
        raise FileNotFoundError(f"no round client metrics found under {run_dir}")
    frames = [pd.read_csv(p) for p in per_round]
    return pd.concat(frames, ignore_index=True)


def _metric_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(np.nanmean(values)), "min": float(np.nanmin(values)), "max": float(np.nanmax(values))}


def _format_cell(stats: Dict[str, float]) -> str:
    if not np.isfinite(stats["mean"]):
        return "nan (nan-nan)"
    return f"{stats['mean']:.4f} ({stats['min']:.4f}-{stats['max']:.4f})"


def _compute_mode_a(df: pd.DataFrame, metrics: List[str]) -> Tuple[Dict[str, Any], int]:
    if "round" not in df.columns:
        raise ValueError("round column missing in round client metrics")
    rounds = sorted(df["round"].dropna().unique().tolist())
    out: Dict[str, Any] = {}
    for m in metrics:
        if m not in df.columns:
            continue
        per_round = df.groupby("round")[m].mean(numeric_only=True)
        stats = _metric_stats(per_round.to_numpy(dtype=float))
        out[m] = stats
    return out, int(len(rounds))


def _compute_mode_b(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in metrics:
        if m not in df.columns:
            continue
        vals = df[m].to_numpy(dtype=float)
        stats = _metric_stats(vals)
        out[m] = stats
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate round x client metrics for FedOcw tables (modes A/B).")
    ap.add_argument("--config", required=True, help="JSON/YAML config with scenarios -> methods -> run_dir mapping.")
    ap.add_argument("--out-dir", default="outputs/fedocw_tables")
    ap.add_argument("--metrics", default="auprc,auroc,brier,nll,ece,accuracy,f1")
    ap.add_argument("--select-mode", choices=["A", "B"], default=None, help="If set, emit a selected summary.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    scenarios: Dict[str, Dict[str, str]] = cfg.get("scenarios", {})
    if not scenarios:
        raise SystemExit("config missing scenarios mapping")
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"metrics": metrics, "scenarios": {}}
    for scenario, methods in scenarios.items():
        meta["scenarios"][scenario] = {}
        for method, run_path in methods.items():
            run_dir = Path(run_path)
            df = _load_round_client_metrics(run_dir)
            stats_a, n_rounds = _compute_mode_a(df, metrics)
            stats_b = _compute_mode_b(df, metrics)
            meta["scenarios"][scenario][method] = {
                "run_dir": str(run_dir),
                "n_rounds": int(n_rounds),
                "n_rows": int(len(df)),
                "mode_a": stats_a,
                "mode_b": stats_b,
            }
            for mode, stats in [("A", stats_a), ("B", stats_b)]:
                for metric, st in stats.items():
                    rows.append(
                        {
                            "scenario": scenario,
                            "method": method,
                            "metric": metric,
                            "mode": mode,
                            "mean": st["mean"],
                            "min": st["min"],
                            "max": st["max"],
                            "formatted": _format_cell(st),
                        }
                    )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "round_summary_all_modes.csv", index=False)
    with (out_dir / "round_summary_all_modes.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    def _write_pivot_tables(df: pd.DataFrame, *, mode: str) -> None:
        df_mode = df[df["mode"] == mode].copy()
        for scenario in sorted(df_mode["scenario"].unique().tolist()):
            df_sc = df_mode[df_mode["scenario"] == scenario]
            pivot = df_sc.pivot(index="method", columns="metric", values="formatted")
            for m in metrics:
                if m not in pivot.columns:
                    pivot[m] = ""
            pivot = pivot.loc[:, metrics]
            out_path = out_dir / f"table_{scenario}_mode{mode}.csv"
            pivot.to_csv(out_path)

    _write_pivot_tables(df_out, mode="A")
    _write_pivot_tables(df_out, mode="B")

    if args.select_mode:
        df_sel = df_out[df_out["mode"] == str(args.select_mode)].copy()
        df_sel.to_csv(out_dir / "round_summary_selected.csv", index=False)
        meta["selected_mode"] = str(args.select_mode)
        with (out_dir / "round_summary_selected.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()

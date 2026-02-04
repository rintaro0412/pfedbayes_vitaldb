from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io import read_json


MethodRow = Dict[str, object]


def _autodetect_run(patterns: Iterable[str]) -> Tuple[Path | None, List[Path]]:
    files: list[Path] = []
    for pat in patterns:
        for p in Path(".").glob(pat):
            if p.is_file():
                files.append(p)
    if not files:
        return None, []
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0].parent, files


def _load_confusion_from_clients(path: Path, *, prefer: str = "pre") -> Dict[str, float] | None:
    if not path.exists():
        return None
    data = read_json(path)
    clients = data.get("clients", {})
    totals = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    found = False
    for v in clients.values():
        conf = v.get(f"confusion_{prefer}")
        if conf is None:
            continue
        found = True
        for k in totals:
            totals[k] += float(conf.get(k, 0.0))
    if not found:
        return None
    denom = sum(totals.values())
    acc = (totals["tp"] + totals["tn"]) / denom if denom > 0 else float("nan")
    totals["accuracy"] = acc
    return totals


def _accuracy_from_confusion(conf: Dict[str, float] | None) -> float | None:
    if conf is None:
        return None
    try:
        tp = float(conf.get("tp", 0.0))
        tn = float(conf.get("tn", 0.0))
        fp = float(conf.get("fp", 0.0))
        fn = float(conf.get("fn", 0.0))
        denom = tp + tn + fp + fn
        if denom <= 0:
            return None
        return (tp + tn) / denom
    except Exception:
        return None


def _load_threshold(run_dir: Path, *, primary: float | None = None, fallback_summary: Path | None = None) -> float | None:
    if primary is not None:
        return float(primary)
    thr_path = run_dir / "threshold.json"
    if thr_path.exists():
        v = read_json(thr_path).get("threshold")
        if v is not None:
            return float(v)
    if fallback_summary is not None and fallback_summary.exists():
        best = read_json(fallback_summary).get("best", {})
        v = best.get("threshold")
        if v is not None:
            return float(v)
    return None


def _load_method_row(method: str, run_dir: Path) -> MethodRow:
    method = method.lower()
    if method == "central":
        report_path = run_dir / "eval_test.json"
        if not report_path.exists():
            raise FileNotFoundError(f"central report not found: {report_path}")
        rep = read_json(report_path)
        metrics = rep.get("metrics_pre")
        if metrics is None:
            raise RuntimeError(f"central report missing metrics: {report_path}")
        conf = rep.get("confusion_pre")
        acc = _accuracy_from_confusion(conf)
        threshold = _load_threshold(run_dir, primary=rep.get("threshold"))
        return {
            "method": "Central",
            "run_dir": run_dir,
            "metrics": metrics,
            "accuracy": acc,
            "threshold": threshold,
            "n": int(rep.get("n", metrics.get("n", 0))),
            "n_pos": int(metrics.get("n_pos", 0)),
            "n_neg": int(metrics.get("n_neg", 0)),
            "artifacts": [report_path],
        }

    if method == "fedavg":
        report_path = run_dir / "test_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"fedavg report not found: {report_path}")
        rep = read_json(report_path)
        metrics = rep.get("metrics_pre")
        if metrics is None:
            raise RuntimeError(f"fedavg report missing metrics: {report_path}")
        conf = rep.get("confusion_pre")
        per_client_conf = _load_confusion_from_clients(run_dir / "test_report_per_client.json", prefer="pre")
        if conf is None:
            conf = per_client_conf
        acc = _accuracy_from_confusion(conf)
        threshold = _load_threshold(run_dir, primary=rep.get("threshold"))
        return {
            "method": "FedAvg",
            "run_dir": run_dir,
            "metrics": metrics,
            "accuracy": acc,
            "threshold": threshold,
            "n": int(rep.get("n", metrics.get("n", 0))),
            "n_pos": int(metrics.get("n_pos", 0)),
            "n_neg": int(metrics.get("n_neg", 0)),
            "artifacts": [report_path, run_dir / "test_report_per_client.json"],
        }

    if method == "fedbe":
        report_path = run_dir / "test_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"fedbe report not found: {report_path}")
        rep = read_json(report_path)
        metrics = rep.get("metrics_pre")
        if metrics is None:
            raise RuntimeError(f"fedbe report missing metrics: {report_path}")
        conf = rep.get("confusion_pre")
        per_client_conf = _load_confusion_from_clients(run_dir / "test_report_per_client.json", prefer="pre")
        if conf is None:
            conf = per_client_conf
        acc = _accuracy_from_confusion(conf)
        threshold = _load_threshold(run_dir, primary=rep.get("threshold"))
        return {
            "method": "FedBE",
            "run_dir": run_dir,
            "metrics": metrics,
            "accuracy": acc,
            "threshold": threshold,
            "n": int(rep.get("n", metrics.get("n", 0))),
            "n_pos": int(metrics.get("n_pos", 0)),
            "n_neg": int(metrics.get("n_neg", 0)),
            "artifacts": [report_path, run_dir / "test_report_per_client.json"],
        }

    if method == "bfl":
        report_path = run_dir / "test_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"bfl report not found: {report_path}")
        rep = read_json(report_path)
        metrics = rep.get("metrics_pre")
        if metrics is None:
            raise RuntimeError(f"bfl report missing metrics: {report_path}")
        conf = rep.get("confusion_pre")
        acc = _accuracy_from_confusion(conf)
        summary_path = run_dir / "summary.json"
        threshold = _load_threshold(run_dir, primary=rep.get("threshold"), fallback_summary=summary_path)
        return {
            "method": "BFL",
            "run_dir": run_dir,
            "metrics": metrics,
            "accuracy": acc,
            "threshold": threshold,
            "n": int(rep.get("n", metrics.get("n", 0))),
            "n_pos": int(metrics.get("n_pos", 0)),
            "n_neg": int(metrics.get("n_neg", 0)),
            "artifacts": [report_path, summary_path],
        }

    raise ValueError(f"unknown method: {method}")


def _load_per_client_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _compute_reliability(y_true: np.ndarray, prob: np.ndarray, *, n_bins: int) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, float]] = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            rows.append({"bin": i, "bin_start": lo, "bin_end": hi, "confidence": np.nan, "accuracy": np.nan, "weight": 0.0})
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask]))
        weight = float(np.mean(mask))
        rows.append({"bin": i, "bin_start": lo, "bin_end": hi, "confidence": conf, "accuracy": acc, "weight": weight})
    return pd.DataFrame(rows)


def _plot_reliability(curves: Dict[str, pd.DataFrame], *, out_png: Path, out_pdf: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    # Define black-and-white friendly styles: (marker, linestyle, color)
    styles = [
        ("o", "-", "black"),  # Circles, solid line, black
        ("^", "--", "dimgray"),  # Triangles, dashed line, gray
        ("s", ":", "darkgray"),  # Squares, dotted line, light gray
    ]

    plt.figure(figsize=(4.0, 4.0), dpi=320)
    plt.plot([0, 1], [0, 1], "--", color="lightgray", label="Ideal")

    # Sort curves by name to ensure consistent styling (e.g., BFL is always the same style)
    sorted_curves = sorted(curves.items())

    for i, (name, df) in enumerate(sorted_curves):
        style = styles[i % len(styles)]
        plt.plot(
            df["confidence"],
            df["accuracy"],
            marker=style[0],
            linestyle=style[1],
            color=style[2],
            label=name,
        )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability (15-bin)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=320)
    if out_pdf is not None:
        plt.savefig(out_pdf)
    plt.close()


def _pos_rate(n_pos: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(n_pos) / float(n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build paper tables/figure from existing evaluation artifacts (no re-training/inference).")
    ap.add_argument("--central-run", default=None, help="Run dir for Central method (expects eval_test.json).")
    ap.add_argument("--fedavg-run", default=None, help="Run dir for FedAvg (expects test_report.json).")
    ap.add_argument("--fedbe-run", default=None, help="Run dir for FedBE (expects test_report.json).")
    ap.add_argument("--bfl-run", default=None, help="Run dir for BFL (expects test_report.json/summary.json).")
    ap.add_argument("--clinical-csv", default="clinical_data.csv", help="Clinical metadata CSV (caseid, department).")
    ap.add_argument("--compare-json", default="runs/compare/bfl_vs_fedavg_test_pre.json", help="Compare JSON for Fig3 (single file).")
    ap.add_argument("--compare-jsons", default=None, help="Comma-separated compare JSONs for Fig3 (merged).")
    ap.add_argument("--data-dir", default="federated_data", help="Windowed NPZ data dir (used only when predictions include case/client ids).")
    ap.add_argument("--out-dir", default=".", help="Output directory for tables/figure.")
    ap.add_argument("--n-bins", type=int, default=15, help="Number of bins for reliability/ECE.")
    ap.add_argument("--allow-missing", action="store_true", help="Do not fail if some methods/artifacts are missing.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect runs when not provided.
    auto_logs: list[str] = []
    if args.central_run is None:
        run, files = _autodetect_run(
            [
                "runs/centralized_batch/*/eval_test.json",
                "runs/centralized/*/eval_test.json",
            ]
        )
        if run is not None:
            args.central_run = str(run)
            picked = files[0]
            auto_logs.append(f"Auto-selected Central run {run} (latest eval_test.json: {picked})")
    if args.fedavg_run is None:
        run, files = _autodetect_run(
            [
                "runs/fedavg_batch/*/test_report.json",
                "runs/fedavg/*/test_report.json",
            ]
        )
        if run is not None:
            args.fedavg_run = str(run)
            picked = files[0]
            auto_logs.append(f"Auto-selected FedAvg run {run} (latest test_report.json: {picked})")
    if args.bfl_run is None:
        run, files = _autodetect_run(
            [
                "runs/bfl_batch/*/test_report.json",
                "runs/bfl/*/test_report.json",
            ]
        )
        if run is not None:
            args.bfl_run = str(run)
            picked = files[0]
            auto_logs.append(f"Auto-selected BFL run {run} (latest test_report.json: {picked})")

    for line in auto_logs:
        print(line)

    missing: list[str] = []
    methods: list[MethodRow] = []
    method_specs = [
        ("central", args.central_run),
        ("fedavg", args.fedavg_run),
        ("bfl", args.bfl_run),
    ]
    if args.fedbe_run is not None:
        method_specs.insert(2, ("fedbe", args.fedbe_run))

    for name, run_dir in method_specs:
        if run_dir is None:
            missing.append(f"{name}: run dir not provided and auto-detection failed")
            continue
        run_path = Path(run_dir)
        if not run_path.exists():
            missing.append(f"{name}: run dir not found: {run_path}")
            continue
        try:
            row = _load_method_row(name, run_path)
            methods.append(row)
            print(f"{row['method']} artifacts: {', '.join([str(p) for p in row.get('artifacts', []) if p is not None])}")
        except Exception as exc:
            missing.append(f"{name}: failed to load metrics ({exc})")

    if not methods:
        print("No methods available; cannot build any table.")
        if missing:
            print("Missing information:")
            for m in missing:
                print(f"- {m}")
        sys.exit(1)

    # Table 2 (global).
    table2_rows: list[dict[str, object]] = []
    for row in methods:
        metrics = row["metrics"]
        acc = row.get("accuracy")
        table2_rows.append(
            {
                "method": row["method"],
                "AUROC": metrics.get("auroc"),
                "AUPRC": metrics.get("auprc"),
                "Accuracy": acc,
                "ECE": metrics.get("ece"),
                "Brier": metrics.get("brier"),
                "NLL": metrics.get("nll"),
                "thr": row.get("threshold"),
                "n_test": row.get("n"),
                "pos_rate": _pos_rate(int(row.get("n_pos", 0)), int(row.get("n", 0))),
            }
        )

    df_table2 = pd.DataFrame(table2_rows)
    table2_path = out_dir / "table2_global.csv"
    df_table2.to_csv(table2_path, index=False)
    print(f"Wrote {table2_path}")
    latex_table2 = df_table2.to_latex(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
    print("\nLaTeX Table 2 (global):\n")
    print(latex_table2)

    # Table 3 (per-client comparison: FedAvg vs BFL/FedBE).
    per_client_data: dict[str, pd.DataFrame] = {}
    for row in methods:
        method = str(row["method"])
        if method not in {"FedAvg", "BFL", "FedBE", "Central"}:
            continue
        run_dir = Path(str(row.get("run_dir", ".")))
        per_client_path = run_dir / "test_report_per_client.csv"
        df = _load_per_client_csv(per_client_path)
        if df is None:
            missing.append(f"Table3: per-client CSV not found for {method}: {per_client_path}")
            continue
        if "client_id" not in df.columns:
            missing.append(f"Table3: per-client CSV missing client_id for {method}: {per_client_path}")
            continue
        per_client_data[method] = df

    pairs: list[tuple[str, str, str]] = []
    if {"FedAvg", "BFL"} <= set(per_client_data.keys()):
        pairs.append(("FedAvg", "BFL", "table3_client.csv"))
    if {"FedAvg", "FedBE"} <= set(per_client_data.keys()):
        out_name = "table3_client_fedbe.csv" if pairs else "table3_client.csv"
        pairs.append(("FedAvg", "FedBE", out_name))

    for method_a, method_b, out_name in pairs:
        df_a = per_client_data[method_a]
        df_b = per_client_data[method_b]
        keep_cols = ["client_id", "n", "n_pos", "auprc_pre", "ece_pre"]
        df_a = df_a[[c for c in keep_cols if c in df_a.columns]].copy()
        df_b = df_b[[c for c in keep_cols if c in df_b.columns]].copy()
        suffix_a = method_a.lower()
        suffix_b = method_b.lower()
        df_a = df_a.rename(
            columns={
                "n": f"n_{suffix_a}",
                "n_pos": f"n_pos_{suffix_a}",
                "auprc_pre": f"auprc_{suffix_a}",
                "ece_pre": f"ece_{suffix_a}",
            }
        )
        df_b = df_b.rename(
            columns={
                "n": f"n_{suffix_b}",
                "n_pos": f"n_pos_{suffix_b}",
                "auprc_pre": f"auprc_{suffix_b}",
                "ece_pre": f"ece_{suffix_b}",
            }
        )
        merged = pd.merge(df_a, df_b, on="client_id", how="inner")
        merged = merged.sort_values("client_id")
        table3_path = out_dir / out_name
        merged.to_csv(table3_path, index=False)
        print(f"Wrote {table3_path}")
        latex_table3 = merged.to_latex(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
        print(f"\nLaTeX Table 3 (per-client: {method_a} vs {method_b}):\n")
        print(latex_table3)

    # Optional Table 3b (per-client comparison: Central vs FedAvg).
    if {"Central", "FedAvg"} <= set(per_client_data.keys()):
        df_c = per_client_data["Central"]
        df_f = per_client_data["FedAvg"]
        keep_cols = ["client_id", "n", "n_pos", "auprc_pre", "ece_pre"]
        df_c = df_c[[c for c in keep_cols if c in df_c.columns]].copy()
        df_f = df_f[[c for c in keep_cols if c in df_f.columns]].copy()
        df_c = df_c.rename(
            columns={
                "n": "n_central",
                "n_pos": "n_pos_central",
                "auprc_pre": "auprc_central",
                "ece_pre": "ece_central",
            }
        )
        df_f = df_f.rename(
            columns={
                "n": "n_fedavg",
                "n_pos": "n_pos_fedavg",
                "auprc_pre": "auprc_fedavg",
                "ece_pre": "ece_fedavg",
            }
        )
        merged_cf = pd.merge(df_c, df_f, on="client_id", how="inner")
        merged_cf = merged_cf.sort_values("client_id")
        table3b_path = out_dir / "table3_central_vs_fedavg.csv"
        merged_cf.to_csv(table3b_path, index=False)
        print(f"Wrote {table3b_path}")
        latex_table3b = merged_cf.to_latex(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
        print("\nLaTeX Table 3b (Central vs FedAvg):\n")
        print(latex_table3b)

    # Fig 3 (reliability diagram) from compare JSON if bins exist.
    # Fig 3 (reliability diagram) from compare JSON(s) if bins exist.
    compare_paths: list[Path] = []
    if args.compare_jsons:
        compare_paths = [Path(p.strip()) for p in str(args.compare_jsons).split(",") if p.strip()]
    else:
        compare_paths = [Path(args.compare_json)]

    curves: dict[str, pd.DataFrame] = {}
    for compare_path in compare_paths:
        if not compare_path.exists():
            missing.append(f"Fig3: compare JSON not found: {compare_path}")
            continue
        compare = read_json(compare_path)
        reliability = compare.get("reliability") or compare.get("calibration_bins")
        if reliability is None:
            missing.append(f"Fig3: compare JSON has no reliability bins: {compare_path}")
            continue
        if isinstance(reliability, dict):
            for key, bins in reliability.items():
                if not isinstance(bins, list):
                    continue
                if str(key) in curves:
                    continue
                df = pd.DataFrame(bins)
                if {"confidence", "accuracy"} <= set(df.columns):
                    curves[str(key)] = df

    if curves:
        if len(curves) < 2:
            missing.append("Fig3: reliability bins missing/insufficient across compare JSONs.")
        else:
            fig_png = out_dir / "fig3_reliability.png"
            fig_pdf = out_dir / "fig3_reliability.pdf"
            _plot_reliability(curves, out_png=fig_png, out_pdf=fig_pdf)
            print(f"Wrote {fig_png} and {fig_pdf}")

    if missing:
        print("\nMissing information:")
        for m in missing:
            print(f"- {m}")
        if not args.allow_missing:
            sys.exit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _load_round_client_metrics(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "round_client_metrics.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df
    # fallback: stitch per-round CSVs
    parts: List[pd.DataFrame] = []
    for f in sorted(run_dir.glob("round_*_test_per_client.csv")):
        try:
            parts.append(pd.read_csv(f))
        except Exception:
            continue
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create boxplots from round x client metrics.")
    ap.add_argument("--runs", required=True, help="Comma-separated run dirs (each contains round_client_metrics.csv).")
    ap.add_argument("--labels", default=None, help="Comma-separated labels (same count as runs).")
    ap.add_argument("--metrics", default="auprc,auroc,brier,nll,ece", help="Comma-separated metric columns.")
    ap.add_argument("--out-dir", default="outputs", help="Output directory.")
    ap.add_argument("--title-prefix", default="Round x Client", help="Figure title prefix.")
    args = ap.parse_args()

    runs = [r.strip() for r in str(args.runs).split(",") if r.strip()]
    labels = [l.strip() for l in str(args.labels).split(",")] if args.labels else []
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]

    if labels and len(labels) != len(runs):
        raise SystemExit("labels count must match runs count")
    if not labels:
        labels = [Path(r).name for r in runs]

    frames: List[pd.DataFrame] = []
    for run, label in zip(runs, labels):
        run_dir = Path(run)
        if not run_dir.exists():
            print(f"[WARN] run dir not found: {run_dir}")
            continue
        df = _load_round_client_metrics(run_dir)
        if df is None:
            print(f"[WARN] no round-client metrics in: {run_dir}")
            continue
        df = df.copy()
        df["method"] = label
        frames.append(df)

    if not frames:
        raise SystemExit("no data found for plotting")

    merged = pd.concat(frames, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "round_client_metrics_merged.csv", index=False)

    # Plot
    import matplotlib.pyplot as plt

    for metric in metrics:
        if metric not in merged.columns:
            print(f"[WARN] missing metric column: {metric}")
            continue
        data = []
        for label in labels:
            vals = merged.loc[merged["method"] == label, metric].dropna().astype(float).values
            data.append(vals)
        if not any(len(v) for v in data):
            print(f"[WARN] no data for metric: {metric}")
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
        )
        ax.set_title(f"{args.title_prefix}: {metric.upper()}")
        ax.set_ylabel(metric.upper())
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        png_path = out_dir / f"boxplot_{metric}.png"
        pdf_path = out_dir / f"boxplot_{metric}.pdf"
        plt.savefig(png_path, dpi=320)
        plt.savefig(pdf_path)
        plt.close()
        print(f"Wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()

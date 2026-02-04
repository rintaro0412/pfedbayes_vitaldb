from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"metrics_round.csv not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _plot(df: pd.DataFrame, *, metrics: List[str], out_dir: Path, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)

    algo_groups = df.groupby("algo")
    colors = {"fedavg": "tab:blue", "fedbe": "tab:orange"}

    for i, metric in enumerate(metrics):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        for algo, g in algo_groups:
            if metric not in g.columns:
                continue
            g = g.sort_values("round")
            ax.plot(
                g["round"],
                g[metric],
                marker="o",
                linewidth=1.5,
                label=str(algo),
                color=colors.get(str(algo).lower(), None),
            )
        ax.set_title(metric.upper())
        ax.set_xlabel("round")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    # hide unused axes
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    png_path = out_dir / "round_metrics.png"
    pdf_path = out_dir / "round_metrics.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot round-wise metrics (AUROC/AUPRC/ECE/NLL/Brier) for FedAvg/FedBE")
    ap.add_argument("--fedavg-run", required=True, help="Run dir containing metrics_round.csv")
    ap.add_argument("--fedbe-run", required=True, help="Run dir containing metrics_round.csv")
    ap.add_argument("--out-dir", default="runs/compare/round_metrics")
    ap.add_argument("--metrics", default="auroc,auprc,ece,nll,brier")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fedavg_path = Path(args.fedavg_run) / "metrics_round.csv"
    fedbe_path = Path(args.fedbe_run) / "metrics_round.csv"

    df_a = _load_metrics(fedavg_path)
    df_b = _load_metrics(fedbe_path)

    df = pd.concat([df_a, df_b], axis=0, ignore_index=True)
    metrics = [m.strip().lower() for m in str(args.metrics).split(",") if m.strip()]

    _plot(df, metrics=metrics, out_dir=Path(args.out_dir), title=args.title)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _prepare(df: pd.DataFrame, *, default_algo: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "round" in out.columns:
        out["round"] = pd.to_numeric(out["round"], errors="coerce")
        out = out.dropna(subset=["round"])
    if "algo" not in out.columns:
        out["algo"] = str(default_algo)
    out["algo"] = out["algo"].astype(str)
    return out


def _pick_metric_frame(*, metric: str, metrics_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    # train_loss is tracked in history.csv for all methods, so prioritize that source.
    candidates = [history_df, metrics_df] if metric == "train_loss" else [metrics_df, history_df]
    for df in candidates:
        if df.empty:
            continue
        if "round" in df.columns and metric in df.columns:
            out = df[["round", "algo", metric]].copy()
            out[metric] = pd.to_numeric(out[metric], errors="coerce")
            out = out.dropna(subset=["round", metric])
            if not out.empty:
                return out
    return pd.DataFrame()


def _plot(
    *,
    runs: List[dict],
    metrics: List[str],
    out_dir: Path,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    colors = {"fedavg": "tab:blue", "fedbe": "tab:orange", "pfedbayes": "tab:green"}

    for i, metric in enumerate(metrics):
        r = i // cols
        c = i % cols
        ax = axes[r][c]

        plotted = False
        for run in runs:
            metric_df = _pick_metric_frame(metric=metric, metrics_df=run["metrics_df"], history_df=run["history_df"])
            if metric_df.empty:
                continue
            for algo, g in metric_df.groupby("algo"):
                g = g.sort_values("round")
                ax.plot(
                    g["round"],
                    g[metric],
                    marker="o",
                    linewidth=1.5,
                    label=str(algo),
                    color=colors.get(str(algo).lower(), None),
                )
                plotted = True

        ax.set_title(metric.upper())
        ax.set_xlabel("round")
        ax.grid(True, linestyle="--", alpha=0.3)
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")

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
    ap = argparse.ArgumentParser(description="Plot round-wise curves from metrics_round.csv/history.csv for two runs.")
    ap.add_argument("--fedavg-run", required=True, help="Run dir containing metrics_round.csv and/or history.csv")
    ap.add_argument("--fedbe-run", required=True, help="Run dir containing metrics_round.csv and/or history.csv")
    ap.add_argument("--out-dir", default="runs/compare/round_metrics")
    ap.add_argument("--metrics", default="train_loss,auroc,auprc,ece,nll,brier")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fedavg_run = Path(args.fedavg_run)
    fedbe_run = Path(args.fedbe_run)

    fedavg_metrics = _prepare(_read_csv_or_empty(fedavg_run / "metrics_round.csv"), default_algo="fedavg")
    fedavg_history = _prepare(_read_csv_or_empty(fedavg_run / "history.csv"), default_algo="fedavg")
    fedbe_metrics = _prepare(_read_csv_or_empty(fedbe_run / "metrics_round.csv"), default_algo="fedbe")
    fedbe_history = _prepare(_read_csv_or_empty(fedbe_run / "history.csv"), default_algo="fedbe")

    if fedavg_metrics.empty and fedavg_history.empty:
        raise FileNotFoundError(f"no metrics_round.csv/history.csv found under {fedavg_run}")
    if fedbe_metrics.empty and fedbe_history.empty:
        raise FileNotFoundError(f"no metrics_round.csv/history.csv found under {fedbe_run}")

    metrics = [m.strip().lower() for m in str(args.metrics).split(",") if m.strip()]
    runs = [
        {"name": "run_a", "metrics_df": fedavg_metrics, "history_df": fedavg_history},
        {"name": "run_b", "metrics_df": fedbe_metrics, "history_df": fedbe_history},
    ]

    _plot(runs=runs, metrics=metrics, out_dir=Path(args.out_dir), title=args.title)


if __name__ == "__main__":
    main()

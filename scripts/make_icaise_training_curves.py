from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodSpec:
    method: str
    run_dir: str
    kind: str


DEFAULT_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("Centralized", "runs/centralized/seed0", "history"),
    MethodSpec("Local", "runs/local/seed42", "local_round_client"),
    MethodSpec("FedAvg", "runs/fedavg/seed42", "history"),
    MethodSpec("FedProx", "runs/fedprox/seed42", "history"),
    MethodSpec("SCAFFOLD", "runs/scaffold/seed42", "history"),
    MethodSpec("FedNova", "runs/fednova/seed42", "history"),
    MethodSpec("Per-FedAvg", "runs/perfedavg/seed42", "history"),
    MethodSpec("pFedMe", "runs/pfedme/seed42", "history"),
    MethodSpec("pFedBayes", "runs/pfedbayes/seed42", "history"),
)


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)


def _load_history(spec: MethodSpec, repo_root: Path) -> pd.DataFrame:
    run_dir = repo_root / spec.run_dir
    if spec.kind == "local_round_client":
        path = run_dir / "round_client_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"empty CSV: {path}")
        if "round" not in df.columns or "train_loss" not in df.columns:
            raise ValueError(f"round/train_loss columns missing: {path}")
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
        if "split" in df.columns:
            # The same train_loss is logged on val/test rows. Keep one split to avoid duplicate weighting.
            preferred = df[df["split"].astype(str).str.lower().eq("val")].copy()
            if not preferred.empty:
                df = preferred
        out = (
            df.groupby("round", as_index=False)
            .agg(
                train_loss=("train_loss", "mean"),
                train_loss_std=("train_loss", "std"),
                n_clients=("client_id", "nunique") if "client_id" in df.columns else ("train_loss", "size"),
            )
            .sort_values("round")
        )
        out["x"] = pd.to_numeric(out["round"], errors="coerce")
        out["source_file"] = str(path)
    else:
        path = run_dir / "history.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        out = pd.read_csv(path)
        if out.empty:
            raise ValueError(f"empty CSV: {path}")
        if "round" in out.columns:
            out["x"] = pd.to_numeric(out["round"], errors="coerce")
        elif "epoch" in out.columns:
            out["x"] = pd.to_numeric(out["epoch"], errors="coerce")
            out["round"] = out["epoch"]
        else:
            raise ValueError(f"round/epoch column missing: {path}")
        out["source_file"] = str(path)

    out["method"] = spec.method
    out["run_dir"] = str(run_dir)
    keep = [
        "method",
        "run_dir",
        "source_file",
        "round",
        "x",
        "train_loss",
        "train_loss_std",
        "n_clients",
        "val_auprc",
        "val_auroc",
        "val_brier",
        "val_nll",
        "val_ece",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep].copy()


def _add_normalized_loss(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["train_loss_norm"] = np.nan
    for method, idx in out.groupby("method").groups.items():
        vals = pd.to_numeric(out.loc[idx, "train_loss"], errors="coerce")
        finite = vals[np.isfinite(vals)]
        if finite.empty:
            continue
        first = float(finite.iloc[0])
        last = float(finite.iloc[-1])
        denom = first - last
        if abs(denom) < 1e-12:
            out.loc[idx, "train_loss_norm"] = vals / max(abs(first), 1e-12)
        else:
            out.loc[idx, "train_loss_norm"] = (vals - last) / denom
    return out


def _plot_lines(
    df: pd.DataFrame,
    *,
    y_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=220)
    for method, g in df.groupby("method", sort=False):
        x = pd.to_numeric(g["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not bool(mask.any()):
            continue
        ax.plot(x[mask], y[mask], linewidth=1.7, label=str(method))
    ax.set_xlabel("Round / epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    _clean_axes(ax)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _parse_method_specs(path: Path | None) -> Sequence[MethodSpec]:
    if path is None:
        return DEFAULT_METHODS
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("methods")
    if not isinstance(data, list):
        raise ValueError("--methods-json must contain a list or an object with methods")
    specs: List[MethodSpec] = []
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("method spec rows must be objects")
        label = row.get("method", row.get("label"))
        run_dir = row.get("run_dir", row.get("source_run_dir"))
        if not label or not run_dir:
            raise ValueError("method spec requires method/label and run_dir/source_run_dir")
        specs.append(MethodSpec(method=str(label), run_dir=str(run_dir), kind=str(row.get("curve_kind", row.get("kind", "history")))))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate ICAISE training curves from existing run logs.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--methods-json", default=None, help="Optional JSON list with method/run_dir/kind entries.")
    ap.add_argument("--out-csv", default="tables/icaise_training_curves.csv")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    methods_json = Path(args.methods_json).resolve() if args.methods_json else None
    specs = _parse_method_specs(methods_json)

    frames: List[pd.DataFrame] = []
    warnings: List[str] = []
    for spec in specs:
        try:
            frames.append(_load_history(spec, repo_root=repo_root))
        except Exception as exc:
            warnings.append(f"{spec.method}: {exc}")
    if not frames:
        raise SystemExit("No training logs could be loaded.")

    curves = _add_normalized_loss(pd.concat(frames, ignore_index=True))
    out_csv = (repo_root / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_csv, index=False)

    figures_dir = (repo_root / args.figures_dir).resolve()
    _plot_lines(
        curves,
        y_col="train_loss",
        ylabel="Training loss (raw)",
        title="Training Loss Curves",
        out_path=figures_dir / "icaise_training_loss_raw.pdf",
        log_y=True,
    )
    _plot_lines(
        curves,
        y_col="train_loss_norm",
        ylabel="Normalized training loss",
        title="Normalized Training Loss Curves",
        out_path=figures_dir / "icaise_training_loss_normalized.pdf",
        log_y=False,
    )
    _plot_lines(
        curves,
        y_col="val_auprc",
        ylabel="Validation AUPRC",
        title="Validation AUPRC Curves",
        out_path=figures_dir / "icaise_validation_auprc.pdf",
        log_y=False,
    )
    _plot_lines(
        curves,
        y_col="val_nll",
        ylabel="Validation NLL",
        title="Validation NLL Curves",
        out_path=figures_dir / "icaise_validation_nll.pdf",
        log_y=False,
    )

    manifest = {
        "csv": str(out_csv),
        "figures": {
            "training_loss_raw": str(figures_dir / "icaise_training_loss_raw.pdf"),
            "training_loss_normalized": str(figures_dir / "icaise_training_loss_normalized.pdf"),
            "validation_auprc": str(figures_dir / "icaise_validation_auprc.pdf"),
            "validation_nll": str(figures_dir / "icaise_validation_nll.pdf"),
        },
        "warnings": warnings,
        "notes": [
            "Centralized epochs are shown on the same horizontal axis as FL communication rounds.",
            "Local training loss is the mean across clients at each round.",
            "pFedBayes raw training loss is a variational objective and is not on the same scale as point-estimate methods.",
        ],
    }
    manifest_path = out_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved curves CSV: {out_csv}")
    print(f"Saved manifest: {manifest_path}")
    for name, path in manifest["figures"].items():
        print(f"Saved {name}: {path}")
    if warnings:
        print("[WARNINGS]")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()

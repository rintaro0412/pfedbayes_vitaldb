"""Create calibration and uncertainty diagnostics from saved pFedBayes predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "entropy": "#0072B2",
    "total_variance": "#009E73",
    "epistemic_variance": "#D55E00",
    "aleatoric_variance": "#CC79A7",
}

LABELS = {
    "entropy": "Predictive entropy",
    "total_variance": "Total variance",
    "epistemic_variance": "Epistemic variance",
    "aleatoric_variance": "Aleatoric variance",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pFedBayes calibration and uncertainty from a prediction NPZ")
    p.add_argument("--pred-npz", default="runs/pfedbayes/seed42/test_predictions.npz")
    p.add_argument("--per-client-json", default="runs/pfedbayes/seed42/test_report_per_client.json")
    p.add_argument("--figure-dir", default="figures/bayesian_evaluation")
    p.add_argument("--table-dir", default="tables/bayesian_evaluation")
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = ["| " + " | ".join(clean(x) for x in headers) + " |"]
    output.append("| " + " | ".join("---" for _ in headers) + " |")
    output.extend("| " + " | ".join(clean(x) for x in row) + " |" for row in rows)
    return "\n".join(output) + "\n"


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)


def calibration_bins(y: np.ndarray, prob: np.ndarray, n_bins: int) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (prob >= lo) & ((prob <= hi) if idx == n_bins - 1 else (prob < hi))
        n = int(mask.sum())
        rows.append(
            {
                "bin": idx + 1,
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "mean_probability": float(prob[mask].mean()) if n else float("nan"),
                "observed_rate": float(y[mask].mean()) if n else float("nan"),
                "gap": float(y[mask].mean() - prob[mask].mean()) if n else float("nan"),
            }
        )
    return rows


def ece(y: np.ndarray, prob: np.ndarray, n_bins: int) -> float:
    total = max(len(y), 1)
    return float(
        sum(row["n"] / total * abs(row["gap"]) for row in calibration_bins(y, prob, n_bins) if row["n"])
    )


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob - y) ** 2))


def nll(y: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(prob, 1e-12, 1.0 - 1e-12)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def auroc_local(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = float(ranks[y == 1].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc_local(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def risk_at_coverage(error: np.ndarray, uncertainty: np.ndarray, coverage: float) -> tuple[float, np.ndarray]:
    keep_n = max(1, int(math.ceil(float(coverage) * len(error))))
    keep = np.argsort(uncertainty, kind="mergesort")[:keep_n]
    return float(error[keep].mean()), keep


def risk_curve(error: np.ndarray, uncertainty: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    coverage = np.linspace(0.05, 1.0, 96)
    risk = np.asarray([risk_at_coverage(error, uncertainty, value)[0] for value in coverage])
    aurc = float(np.trapz(risk, coverage) / (coverage[-1] - coverage[0]))
    return coverage, risk, aurc


def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def short_client(name: str) -> str:
    return name.replace("General_surgery__", "GS / ").replace("Thoracic_surgery__", "TS / ").replace("_", " ")


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_npz)
    client_path = Path(args.per_client_json)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    with np.load(pred_path, allow_pickle=False) as data:
        required = {"y_true", "prob_mean", "prob_epi", "prob_alea", "prob_total_var", "entropy", "case_id"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing prediction keys: {sorted(missing)}")
        y = np.asarray(data["y_true"], dtype=np.int64).reshape(-1)
        prob = np.asarray(data["prob_mean"], dtype=np.float64).reshape(-1)
        case_id = np.asarray(data["case_id"]).reshape(-1)
        measures = {
            "entropy": np.asarray(data["entropy"], dtype=np.float64).reshape(-1),
            "total_variance": np.asarray(data["prob_total_var"], dtype=np.float64).reshape(-1),
            "epistemic_variance": np.asarray(data["prob_epi"], dtype=np.float64).reshape(-1),
            "aleatoric_variance": np.asarray(data["prob_alea"], dtype=np.float64).reshape(-1),
        }

    if not all(len(value) == len(y) for value in [prob, case_id, *measures.values()]):
        raise ValueError("Prediction arrays have inconsistent lengths")
    finite = np.isfinite(prob)
    for value in measures.values():
        finite &= np.isfinite(value)
    y, prob, case_id = y[finite], np.clip(prob[finite], 1e-12, 1.0 - 1e-12), case_id[finite]
    measures = {key: value[finite] for key, value in measures.items()}
    error = ((prob >= 0.5).astype(np.int64) != y).astype(np.int64)

    cal_rows = calibration_bins(y, prob, args.n_bins)
    point_metrics = {"ECE (15 equal-width bins)": ece(y, prob, args.n_bins), "Brier score": brier(y, prob), "NLL": nll(y, prob)}

    groups = np.unique(case_id)
    group_rows = [np.flatnonzero(case_id == group) for group in groups]
    rng = np.random.default_rng(args.seed)
    boot_cal = {key: [] for key in point_metrics}
    boot_unc = {key: {"error_auroc": [], "risk_80": []} for key in measures}
    for _ in range(args.bootstrap):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        index = np.concatenate([group_rows[group_idx] for group_idx in sampled_groups])
        y_b, p_b, e_b = y[index], prob[index], error[index]
        boot_cal["ECE (15 equal-width bins)"].append(ece(y_b, p_b, args.n_bins))
        boot_cal["Brier score"].append(brier(y_b, p_b))
        boot_cal["NLL"].append(nll(y_b, p_b))
        for key, uncertainty in measures.items():
            u_b = uncertainty[index]
            boot_unc[key]["error_auroc"].append(auroc_local(e_b, u_b))
            boot_unc[key]["risk_80"].append(risk_at_coverage(e_b, u_b, 0.8)[0])

    metric_table_rows = []
    for key, estimate in point_metrics.items():
        low, high = ci(np.asarray(boot_cal[key]))
        metric_table_rows.append([key, f"{estimate:.4f}", f"[{low:.4f}, {high:.4f}]"])
    (table_dir / "table11_calibration_summary.md").write_text(
        "# pFedBayes calibration summary\n\n"
        + f"Test set: {len(y):,} windows, {len(groups):,} cases. Intervals use {args.bootstrap} case-cluster bootstrap replicates (seed {args.seed}).\n\n"
        + markdown_table(["Metric", "Estimate", "95% interval"], metric_table_rows),
        encoding="utf-8",
    )

    bin_table_rows = [
        [
            row["bin"], f"[{row['lo']:.2f}, {row['hi']:.2f}{']' if row['bin'] == args.n_bins else ')'}",
            f"{int(row['n']):,}", "—" if not row["n"] else f"{row['mean_probability']:.4f}",
            "—" if not row["n"] else f"{row['observed_rate']:.4f}", "—" if not row["n"] else f"{row['gap']:+.4f}",
        ]
        for row in cal_rows
    ]
    (table_dir / "table12_reliability_bins.md").write_text(
        "# Reliability bins\n\n"
        + markdown_table(["Bin", "Probability range", "n", "Mean probability", "Observed positive rate", "Observed − predicted"], bin_table_rows),
        encoding="utf-8",
    )

    utility_rows: list[list[object]] = []
    risk_rows: list[list[object]] = []
    utility_payload: dict[str, dict[str, float | list[float]]] = {}
    selected_coverages = [1.0, 0.9, 0.8, 0.7, 0.5]
    for key, uncertainty in measures.items():
        err_auroc = auroc_local(error, uncertainty)
        err_auprc = auprc_local(error, uncertainty)
        risk80, keep80 = risk_at_coverage(error, uncertainty, 0.8)
        coverage, risk, aurc = risk_curve(error, uncertainty)
        auroc_ci = ci(np.asarray(boot_unc[key]["error_auroc"]))
        risk_ci = ci(np.asarray(boot_unc[key]["risk_80"]))
        utility_rows.append(
            [
                LABELS[key], f"{err_auroc:.4f} [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]", f"{err_auprc:.4f}",
                f"{uncertainty[error == 0].mean():.5f}", f"{uncertainty[error == 1].mean():.5f}",
                f"{risk80:.4f} [{risk_ci[0]:.4f}, {risk_ci[1]:.4f}]", f"{aurc:.4f}",
            ]
        )
        for retained in selected_coverages:
            retained_risk, keep = risk_at_coverage(error, uncertainty, retained)
            risk_rows.append([LABELS[key], f"{retained:.0%}", f"{retained_risk:.4f}", f"{y[keep].mean():.4f}", f"{len(keep):,}"])
        utility_payload[key] = {
            "error_auroc": err_auroc,
            "error_auprc": err_auprc,
            "risk_at_80": risk80,
            "positive_rate_at_80": float(y[keep80].mean()),
            "aurc_coverage_0.05_to_1": aurc,
            "error_auroc_ci": list(auroc_ci),
            "risk_at_80_ci": list(risk_ci),
        }
    (table_dir / "table13_uncertainty_utility.md").write_text(
        "# Uncertainty utility for detecting prediction errors\n\n"
        + "Higher uncertainty is treated as a score for an incorrect threshold-0.5 prediction. Error prevalence is "
        + f"{error.mean():.4f}. AUPRC should therefore be compared with {error.mean():.4f}.\n\n"
        + markdown_table(
            ["Measure", "Error AUROC (95% CI)", "Error AUPRC", "Mean if correct", "Mean if error", "Risk at 80% coverage (95% CI)", "AURC"],
            utility_rows,
        ),
        encoding="utf-8",
    )
    (table_dir / "table14_risk_coverage.md").write_text(
        "# Selected risk–coverage results\n\n"
        + markdown_table(["Uncertainty", "Coverage", "Error rate", "Positive-label rate retained", "n retained"], risk_rows),
        encoding="utf-8",
    )

    client_report = json.loads(client_path.read_text(encoding="utf-8"))
    client_rows = []
    for client, report in client_report["clients"].items():
        if report.get("status") != "ok":
            continue
        metrics = report["metrics_pre"]
        client_rows.append(
            {
                "client": client,
                "n": int(metrics["n"]),
                "positive_rate": int(metrics["n_pos"]) / max(int(metrics["n"]), 1),
                "ece": float(metrics["ece"]),
                "brier": float(metrics["brier"]),
                "auprc": float(metrics["auprc"]),
                "auroc": float(metrics["auroc"]),
            }
        )
    client_rows.sort(key=lambda row: row["ece"])
    (table_dir / "table15_client_calibration.md").write_text(
        "# Per-client calibration\n\n"
        + "Clients are surgery/procedure pseudo-clients, not independent hospitals. ECE is descriptive and has no interval here.\n\n"
        + markdown_table(
            ["Pseudo-client", "n", "Positive rate", "ECE", "Brier", "AUPRC", "AUROC"],
            [[short_client(str(row["client"])), f"{row['n']:,}", f"{row['positive_rate']:.3f}", f"{row['ece']:.4f}", f"{row['brier']:.4f}", f"{row['auprc']:.4f}", f"{row['auroc']:.4f}"] for row in client_rows],
        ),
        encoding="utf-8",
    )

    fig = plt.figure(figsize=(8.8, 7.2), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    valid = [row for row in cal_rows if row["n"]]
    sizes = np.asarray([row["n"] for row in valid], dtype=float)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="Perfect calibration")
    ax.plot([row["mean_probability"] for row in valid], [row["observed_rate"] for row in valid], marker="o", color=COLORS["entropy"], linewidth=2, label="pFedBayes")
    ax.scatter([row["mean_probability"] for row in valid], [row["observed_rate"] for row in valid], s=30 + 100 * sizes / sizes.max(), color=COLORS["entropy"])
    ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed positive rate", title=f"pFedBayes reliability diagram (ECE={point_metrics['ECE (15 equal-width bins)']:.3f})")
    ax.legend(frameon=False)
    style_axis(ax)
    ax_hist = fig.add_subplot(gs[1], sharex=ax)
    ax_hist.hist(prob, bins=np.linspace(0, 1, args.n_bins + 1), color="#56B4E9", edgecolor="white")
    ax_hist.set(xlabel="Predicted probability", ylabel="Count")
    style_axis(ax_hist)
    save_figure(fig, figure_dir, "fig13_pfedbayes_reliability")

    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    x = np.arange(1, 11)
    for key in ["entropy", "epistemic_variance", "aleatoric_variance"]:
        order = np.argsort(measures[key], kind="mergesort")
        chunks = np.array_split(order, 10)
        rates = [float(error[chunk].mean()) for chunk in chunks]
        ax.plot(x, rates, marker="o", linewidth=2, label=LABELS[key], color=COLORS[key])
    ax.axhline(error.mean(), color="0.4", linestyle="--", linewidth=1.2, label="Overall error rate")
    ax.set(xticks=x, xlabel="Uncertainty decile (1 = least uncertain)", ylabel="Prediction error rate", title="Does uncertainty concentrate prediction errors?")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, figure_dir, "fig14_uncertainty_error_deciles")

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for key in ["entropy", "epistemic_variance", "aleatoric_variance"]:
        coverage, risk, _ = risk_curve(error, measures[key])
        ax.plot(coverage, risk, linewidth=2, label=LABELS[key], color=COLORS[key])
    ax.axhline(error.mean(), color="0.4", linestyle="--", linewidth=1.2, label="No rejection")
    ax.set(xlim=(0.5, 1.0), xlabel="Coverage retained (least uncertain first)", ylabel="Prediction error rate", title="pFedBayes risk–coverage comparison")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, figure_dir, "fig15_uncertainty_risk_coverage")

    fig, ax = plt.subplots(figsize=(10.0, 8.8))
    y_pos = np.arange(len(client_rows))
    eces = [float(row["ece"]) for row in client_rows]
    ax.barh(y_pos, eces, color=["#56B4E9" if value <= point_metrics["ECE (15 equal-width bins)"] else "#E69F00" for value in eces])
    ax.axvline(point_metrics["ECE (15 equal-width bins)"], color="black", linestyle="--", linewidth=1.2, label="Global ECE")
    ax.set(yticks=y_pos, yticklabels=[short_client(str(row["client"])) for row in client_rows], xlabel="ECE (15 bins)", title="Calibration varies across surgery/procedure pseudo-clients")
    ax.legend(frameon=False, loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, figure_dir, "fig16_client_calibration_ece")

    total_identity_error = float(np.max(np.abs(measures["total_variance"] - prob * (1.0 - prob))))
    metadata = {
        "prediction_file": str(pred_path),
        "per_client_file": str(client_path),
        "n_windows": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_cases": int(len(groups)),
        "threshold": 0.5,
        "n_bins": int(args.n_bins),
        "bootstrap_replicates": int(args.bootstrap),
        "bootstrap_unit": "case_id",
        "seed": int(args.seed),
        "calibration": point_metrics,
        "error_rate": float(error.mean()),
        "uncertainty_utility": utility_payload,
        "max_abs_total_variance_minus_p_times_one_minus_p": total_identity_error,
        "note": "No post-hoc calibration was fitted on the test set. Total variance equals p_mean*(1-p_mean) up to numerical precision under the current decomposition.",
    }
    (figure_dir / "bayesian_evaluation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

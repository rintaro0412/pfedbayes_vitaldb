from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.metrics import auprc, auroc, brier_score, expected_calibration_error, nll_binary


def _parse_coverages(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if val > 1.0:
            val = val / 100.0
        if not (0.0 < val <= 1.0):
            raise argparse.ArgumentTypeError("coverage values must be in (0, 1] or percent in (0, 100]")
        vals.append(val)
    if not vals:
        raise argparse.ArgumentTypeError("at least one coverage value is required")
    return sorted(set(vals), reverse=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _metric_or_nan(fn: Callable[..., float], y: np.ndarray, p: np.ndarray, **kwargs: Any) -> float:
    try:
        val = float(fn(y, p, **kwargs))
    except Exception:
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def _auprc_local(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def _auroc_local(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i + 1
        while j < len(sorted_p) and sorted_p[j] == sorted_p[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auprc_value(y_true: np.ndarray, prob: np.ndarray) -> float:
    val = _metric_or_nan(auprc, y_true, prob)
    return val if math.isfinite(val) else _auprc_local(y_true, prob)


def _auroc_value(y_true: np.ndarray, prob: np.ndarray) -> float:
    val = _metric_or_nan(auroc, y_true, prob)
    return val if math.isfinite(val) else _auroc_local(y_true, prob)


def _select_uncertainty_key(files: Iterable[str], preferred: str | None) -> str:
    keys = set(files)
    if preferred:
        if preferred not in keys:
            raise KeyError(f"uncertainty key not found: {preferred}")
        return preferred
    for key in ("prob_total_var", "prob_var", "prob_epi", "entropy"):
        if key in keys:
            return key
    raise KeyError("no uncertainty key found; expected one of prob_total_var, prob_var, prob_epi, entropy")


def _compute_rows(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    uncertainty: np.ndarray,
    coverages: List[float],
    n_bins: int,
) -> List[Dict[str, Any]]:
    n = int(y_true.shape[0])
    order = np.argsort(uncertainty, kind="mergesort")
    rows: List[Dict[str, Any]] = []
    for coverage in coverages:
        keep_n = max(1, int(math.ceil(float(coverage) * n)))
        keep_idx = order[:keep_n]
        y = y_true[keep_idx]
        p = prob[keep_idx]
        u = uncertainty[keep_idx]
        pred = (p >= 0.5).astype(np.int64)
        error = float(np.mean(pred != y))
        rows.append(
            {
                "coverage": float(keep_n / n),
                "requested_coverage": float(coverage),
                "n_retained": keep_n,
                "n_total": n,
                "n_pos_retained": int(np.sum(y == 1)),
                "n_neg_retained": int(np.sum(y == 0)),
                "uncertainty_max_retained": float(np.max(u)),
                "uncertainty_mean_retained": float(np.mean(u)),
                "risk_error_rate": error,
                "accuracy": float(1.0 - error),
                "auprc": _auprc_value(y, p),
                "auroc": _auroc_value(y, p),
                "brier": _metric_or_nan(brier_score, y, p),
                "nll": _metric_or_nan(nll_binary, y, p),
                "ece": _metric_or_nan(expected_calibration_error, y, p, n_bins=int(n_bins)),
            }
        )
    return rows


def _write_plot(rows: List[Dict[str, Any]], out_dir: Path, *, metric: str) -> str | None:
    xs = np.asarray([float(r["coverage"]) for r in rows], dtype=float)
    ys = np.asarray([float(r[metric]) for r in rows], dtype=float)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    if plt is not None:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(xs, ys, marker="o", linewidth=1.8)
        ax.set_xlabel("Coverage")
        ax.set_ylabel(metric)
        ax.set_title(f"Risk-coverage curve ({metric})")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(max(0.0, float(np.nanmin(xs)) - 0.02), 1.01)
        fig.tight_layout()
        out_path = out_dir / f"risk_coverage_{metric}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return str(out_path)

    # Minimal dependency-free SVG fallback for environments without matplotlib.
    width, height = 720, 460
    left, right, top, bottom = 70, 30, 45, 65
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    if math.isclose(x_min, x_max):
        x_min, x_max = x_min - 0.01, x_max + 0.01
    if math.isclose(y_min, y_max):
        y_min, y_max = y_min - 0.01, y_max + 0.01
    y_pad = 0.05 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * (width - left - right)

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * (height - top - bottom)

    points = " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(xs, ys))
    circles = "\n".join(
        f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" r="4" fill="#1f77b4" />'
        for x, y in zip(xs, ys)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Risk-coverage curve ({metric})</text>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#222" stroke-width="1"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#222" stroke-width="1"/>
  <text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="14">Coverage</text>
  <text x="18" y="{height / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 18 {height / 2:.1f})">{metric}</text>
  <text x="{left}" y="{height - bottom + 22}" text-anchor="middle" font-family="Arial" font-size="12">{x_min:.2f}</text>
  <text x="{width - right}" y="{height - bottom + 22}" text-anchor="middle" font-family="Arial" font-size="12">{x_max:.2f}</text>
  <text x="{left - 8}" y="{height - bottom + 4}" text-anchor="end" font-family="Arial" font-size="12">{y_min:.3f}</text>
  <text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="12">{y_max:.3f}</text>
  <polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="3"/>
  {circles}
</svg>
'''
    out_path = out_dir / f"risk_coverage_{metric}.svg"
    out_path.write_text(svg, encoding="utf-8")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create risk-coverage tables and plots from saved prediction NPZ.")
    ap.add_argument("--pred-npz", required=True, help="Prediction NPZ, e.g. runs/pfedbayes/seed42/test_predictions.npz")
    ap.add_argument("--out-dir", default="outputs_plan/risk_coverage_pfedbayes")
    ap.add_argument("--prob-key", default="prob_mean")
    ap.add_argument("--y-key", default="y_true")
    ap.add_argument("--uncertainty-key", default=None, help="Default: prob_total_var, then prob_var, prob_epi, entropy.")
    ap.add_argument(
        "--coverages",
        default="1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5",
        help="Comma-separated fractions or percentages. Retains the least uncertain samples.",
    )
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--plot-metric", default="risk_error_rate", choices=["risk_error_rate", "auprc", "auroc", "brier", "nll", "ece"])
    args = ap.parse_args()

    pred_path = Path(args.pred_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(pred_path, allow_pickle=True)
    uncertainty_key = _select_uncertainty_key(data.files, args.uncertainty_key)

    y_true = np.asarray(data[str(args.y_key)], dtype=np.int64).reshape(-1)
    prob = np.asarray(data[str(args.prob_key)], dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(data[uncertainty_key], dtype=np.float64).reshape(-1)

    if not (y_true.shape == prob.shape == uncertainty.shape):
        raise SystemExit(f"shape mismatch: y={y_true.shape}, prob={prob.shape}, uncertainty={uncertainty.shape}")

    finite = np.isfinite(prob) & np.isfinite(uncertainty)
    if not np.all(finite):
        y_true = y_true[finite]
        prob = prob[finite]
        uncertainty = uncertainty[finite]

    rows = _compute_rows(
        y_true=y_true,
        prob=np.clip(prob, 1e-7, 1.0 - 1e-7),
        uncertainty=uncertainty,
        coverages=_parse_coverages(str(args.coverages)),
        n_bins=int(args.n_bins),
    )

    csv_path = out_dir / "risk_coverage.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = _write_plot(rows, out_dir, metric=str(args.plot_metric))
    payload = {
        "pred_npz": str(pred_path),
        "prob_key": str(args.prob_key),
        "y_key": str(args.y_key),
        "uncertainty_key": uncertainty_key,
        "n": int(y_true.shape[0]),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
        "rows": rows,
        "csv": str(csv_path),
        "plot": plot_path,
    }
    json_path = out_dir / "risk_coverage.json"
    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    if plot_path:
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()

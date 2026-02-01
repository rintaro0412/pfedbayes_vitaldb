from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = 1.0 / (1.0 + np.exp(-x))
    return out.astype(np.float64, copy=False)


def brier_score(y_true: np.ndarray, prob: np.ndarray, *, sample_weight: np.ndarray | None = None) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    err = (p - y) ** 2
    if sample_weight is None:
        return float(np.mean(err))
    w = np.asarray(sample_weight, dtype=np.float64)
    return float(np.average(err, weights=w))


def nll_binary(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    eps: float = 1e-12,
    sample_weight: np.ndarray | None = None,
) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if sample_weight is None:
        return float(np.mean(loss))
    w = np.asarray(sample_weight, dtype=np.float64)
    return float(np.average(loss, weights=w))


def expected_calibration_error(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    n_bins: int = 15,
    sample_weight: np.ndarray | None = None,
) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    if sample_weight is None:
        w = np.ones_like(y, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.shape != y.shape:
            raise ValueError(f"sample_weight shape mismatch: {w.shape} != {y.shape}")
    w_total = float(np.sum(w))
    if w_total <= 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        w_bin = float(np.sum(w[m]))
        if w_bin <= 0:
            continue
        acc = float(np.average(y[m], weights=w[m]))
        conf = float(np.average(p[m], weights=w[m]))
        ece += (w_bin / w_total) * abs(acc - conf)
    return float(ece)


def _safe_sklearn_metric(
    fn_name: str,
    y: np.ndarray,
    p: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        y = np.asarray(y, dtype=np.int64)
        p = np.asarray(p, dtype=np.float64)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if fn_name == "auprc":
            return float(average_precision_score(y, p, sample_weight=w))
        if fn_name == "auroc":
            return float(roc_auc_score(y, p, sample_weight=w))
        raise ValueError(fn_name)
    except Exception:
        return float("nan")


def auprc(y_true: np.ndarray, prob: np.ndarray, *, sample_weight: np.ndarray | None = None) -> float:
    return _safe_sklearn_metric("auprc", y_true, prob, sample_weight=sample_weight)


def auroc(y_true: np.ndarray, prob: np.ndarray, *, sample_weight: np.ndarray | None = None) -> float:
    return _safe_sklearn_metric("auroc", y_true, prob, sample_weight=sample_weight)


def confusion_at_threshold(y_true: np.ndarray, prob: np.ndarray, *, thr: float) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob, dtype=np.float64)
    pred = (p >= float(thr)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(acc),
    }


def derived_from_confusion(conf: Dict[str, float]) -> Dict[str, float]:
    tp = float(conf.get("tp", 0.0))
    tn = float(conf.get("tn", 0.0))
    fp = float(conf.get("fp", 0.0))
    fn = float(conf.get("fn", 0.0))
    denom_f1 = (2 * tp + fp + fn)
    f1 = (2 * tp / denom_f1) if denom_f1 > 0 else float("nan")
    denom_mcc = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_mcc <= 0:
        mcc = float("nan")
    else:
        mcc = (tp * tn - fp * fn) / float(np.sqrt(denom_mcc))
    return {
        "accuracy": float(conf.get("accuracy", float("nan"))),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def best_threshold_f1(y_true: np.ndarray, prob: np.ndarray, *, grid: int = 101) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob, dtype=np.float64)
    thrs = np.linspace(0.0, 1.0, int(grid))
    best_thr = 0.5
    best_f1 = -1.0
    for t in thrs:
        pred = p >= float(t)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(t)
    return float(best_thr)


def best_threshold_youden(y_true: np.ndarray, prob: np.ndarray, *, fallback: float = 0.5) -> float:
    try:
        from sklearn.metrics import roc_curve

        y = np.asarray(y_true, dtype=np.int64)
        p = np.asarray(prob, dtype=np.float64)
        fpr, tpr, thr = roc_curve(y, p)
        j = tpr - fpr
        idx = int(np.nanargmax(j))
        return float(thr[idx])
    except Exception:
        return float(fallback)


@dataclass(frozen=True)
class BinaryMetrics:
    n: int
    n_pos: int
    n_neg: int
    auprc: float
    auroc: float
    brier: float
    nll: float
    ece: float


def compute_binary_metrics(y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 15) -> BinaryMetrics:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob, dtype=np.float64)
    return BinaryMetrics(
        n=int(y.shape[0]),
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
        auprc=auprc(y, p),
        auroc=auroc(y, p),
        brier=brier_score(y, p),
        nll=nll_binary(y, p),
        ece=expected_calibration_error(y, p, n_bins=n_bins),
    )


def bootstrap_group_ci(
    *,
    group_ids: np.ndarray,
    y_true: np.ndarray,
    prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Patient/group bootstrap CI on *window-level* labels/preds by resampling groups with replacement.
    """
    g = np.asarray(group_ids)
    y = np.asarray(y_true)
    p = np.asarray(prob)
    uniq, inv = np.unique(g, return_inverse=True)
    n_groups = int(uniq.size)
    if n_groups == 0:
        return {"mean": float("nan"), "p2_5": float("nan"), "p97_5": float("nan")}

    rng = np.random.RandomState(int(seed))
    vals = []
    idx_by_group = [np.where(inv == i)[0] for i in range(n_groups)]
    for _ in range(int(n_boot)):
        sampled = rng.choice(n_groups, size=n_groups, replace=True)
        counts = np.bincount(sampled, minlength=n_groups).astype(np.float64)
        w = counts[inv]
        try:
            vals.append(float(metric_fn(y, p, sample_weight=w)))
            continue
        except TypeError:
            pass
        except Exception:
            vals.append(float("nan"))
            continue

        # fallback: materialize duplicated indices
        try:
            idx = np.concatenate([idx_by_group[int(i)] for i in sampled], axis=0)
            vals.append(float(metric_fn(y[idx], p[idx])))
        except Exception:
            vals.append(float("nan"))

    arr = np.asarray(vals, dtype=np.float64)
    mean = float(np.nanmean(arr))
    lo = float(np.nanpercentile(arr, 2.5))
    hi = float(np.nanpercentile(arr, 97.5))
    return {"mean": mean, "p2_5": lo, "p97_5": hi}


def bootstrap_group_diff_ci(
    *,
    group_ids: np.ndarray,
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired group bootstrap CI for (metric_b - metric_a) by resampling groups with replacement.

    Uses group-level weights so repeated groups contribute multiple times (equivalent to duplicating groups).
    """
    g = np.asarray(group_ids)
    y = np.asarray(y_true)
    a = np.asarray(prob_a)
    b = np.asarray(prob_b)
    if y.shape != a.shape or y.shape != b.shape or y.shape != g.shape:
        raise ValueError(f"shape mismatch: y={y.shape}, a={a.shape}, b={b.shape}, g={g.shape}")

    uniq, inv = np.unique(g, return_inverse=True)
    n_groups = int(uniq.size)
    if n_groups == 0:
        return {
            "a": float("nan"),
            "b": float("nan"),
            "diff": float("nan"),
            "diff_p2_5": float("nan"),
            "diff_p97_5": float("nan"),
            "p_two_sided": float("nan"),
            "p_greater": float("nan"),
            "p_less": float("nan"),
            "n_boot": float(n_boot),
            "n_valid": 0.0,
        }

    rng = np.random.RandomState(int(seed))
    diffs = []
    idx_by_group = [np.where(inv == i)[0] for i in range(n_groups)]
    for _ in range(int(n_boot)):
        sampled = rng.choice(n_groups, size=n_groups, replace=True)
        counts = np.bincount(sampled, minlength=n_groups).astype(np.float64)
        w = counts[inv]
        try:
            m_a = float(metric_fn(y, a, sample_weight=w))
            m_b = float(metric_fn(y, b, sample_weight=w))
        except TypeError:
            # fallback: materialize duplicated indices if metric_fn doesn't accept sample_weight
            idx = np.concatenate([idx_by_group[int(i)] for i in sampled], axis=0)
            m_a = float(metric_fn(y[idx], a[idx]))
            m_b = float(metric_fn(y[idx], b[idx]))
        except Exception:
            diffs.append(float("nan"))
            continue
        diffs.append(float(m_b - m_a))

    arr = np.asarray(diffs, dtype=np.float64)
    finite = np.isfinite(arr)
    n_valid = int(np.sum(finite))
    if n_valid == 0:
        return {
            "a": float(metric_fn(y, a)),
            "b": float(metric_fn(y, b)),
            "diff": float("nan"),
            "diff_p2_5": float("nan"),
            "diff_p97_5": float("nan"),
            "p_two_sided": float("nan"),
            "p_greater": float("nan"),
            "p_less": float("nan"),
            "n_boot": float(n_boot),
            "n_valid": 0.0,
        }

    diff_mean = float(np.nanmean(arr))
    lo = float(np.nanpercentile(arr, 2.5))
    hi = float(np.nanpercentile(arr, 97.5))

    arr_f = arr[finite]
    # Add-one smoothing to avoid 0 p-values
    p_greater = float((np.sum(arr_f <= 0.0) + 1.0) / (float(n_valid) + 1.0))  # H1: diff > 0
    p_less = float((np.sum(arr_f >= 0.0) + 1.0) / (float(n_valid) + 1.0))     # H1: diff < 0
    p_two = float(min(1.0, 2.0 * min(p_greater, p_less)))

    return {
        "a": float(metric_fn(y, a)),
        "b": float(metric_fn(y, b)),
        "diff": float(diff_mean),
        "diff_p2_5": float(lo),
        "diff_p97_5": float(hi),
        "p_two_sided": float(p_two),
        "p_greater": float(p_greater),
        "p_less": float(p_less),
        "n_boot": float(n_boot),
        "n_valid": float(n_valid),
    }

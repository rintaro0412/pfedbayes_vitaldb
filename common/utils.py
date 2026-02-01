from __future__ import annotations

from typing import Dict

import numpy as np

from common.experiment import seed_everything
from common.metrics import compute_binary_metrics, confusion_at_threshold, sigmoid_np


def set_seed(seed: int) -> None:
    seed_everything(seed, deterministic=True)


def calc_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
    n_bins: int = 15,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    if from_logits:
        p = sigmoid_np(p)
    base = compute_binary_metrics(y, p, n_bins=int(n_bins))
    conf = confusion_at_threshold(y, p, thr=float(threshold))
    tp = conf.get("tp", 0.0)
    fp = conf.get("fp", 0.0)
    fn = conf.get("fn", 0.0)
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else float("nan")
    return {
        "n": float(base.n),
        "n_pos": float(base.n_pos),
        "n_neg": float(base.n_neg),
        "pos_rate": float(base.n_pos / max(base.n, 1)),
        "auprc": float(base.auprc),
        "auroc": float(base.auroc),
        "brier": float(base.brier),
        "nll": float(base.nll),
        "ece": float(base.ece),
        "threshold": float(threshold),
        "f1": float(f1),
        "sensitivity": float(conf.get("sensitivity", float("nan"))),
        "specificity": float(conf.get("specificity", float("nan"))),
        "ppv": float(conf.get("ppv", float("nan"))),
        "npv": float(conf.get("npv", float("nan"))),
        "accuracy": float(conf.get("accuracy", float("nan"))),
        "tp": float(conf.get("tp", float("nan"))),
        "tn": float(conf.get("tn", float("nan"))),
        "fp": float(conf.get("fp", float("nan"))),
        "fn": float(conf.get("fn", float("nan"))),
    }

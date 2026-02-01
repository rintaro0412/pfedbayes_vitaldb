from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        raise ValueError(f"temperature must be finite and >0, got {temperature}")
    return np.asarray(logits, dtype=np.float64) / t


@dataclass(frozen=True)
class TemperatureFitResult:
    temperature: float
    nll_before: float
    nll_after: float


def fit_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    device: str = "cpu",
    max_iter: int = 100,
) -> TemperatureFitResult:
    """
    Temperature scaling for binary classification.

    Fits T>0 minimizing BCEWithLogitsLoss(logits/T, y).
    """
    import torch

    x = np.asarray(logits, dtype=np.float32).reshape(-1, 1)
    y = np.asarray(y_true, dtype=np.float32).reshape(-1, 1)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        nll_before = float(loss_fn(x_t, y_t).item())

    log_t = torch.nn.Parameter(torch.zeros(1, device=device))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=int(max_iter), line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        t = torch.exp(log_t).clamp(1e-3, 1e3)
        loss = loss_fn(x_t / t, y_t)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        t = float(torch.exp(log_t).clamp(1e-3, 1e3).item())
        nll_after = float(loss_fn(x_t / t, y_t).item())

    return TemperatureFitResult(temperature=t, nll_before=nll_before, nll_after=nll_after)


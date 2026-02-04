from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

StateDict = Dict[str, torch.Tensor]


def fedavg_state(client_states: Sequence[StateDict], weights: Sequence[float] | None = None) -> StateDict:
    if not client_states:
        raise ValueError("client_states is empty")
    if weights is None:
        weights = [1.0] * len(client_states)
    if len(weights) != len(client_states):
        raise ValueError("weights length must match client_states")
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    avg: StateDict = {}
    for k in client_states[0].keys():
        acc = torch.zeros_like(client_states[0][k])
        for st, wk in zip(client_states, w):
            acc = acc + st[k] * float(wk)
        avg[k] = acc
    return avg


def _compute_mean_sq(
    *,
    base_state: StateDict,
    teacher_states: Sequence[StateDict],
    eps: float = 1e-12,
) -> tuple[StateDict, StateDict, Dict[str, torch.Tensor]]:
    w_avg: StateDict = {}
    w_sq_avg: StateDict = {}
    w_norm: Dict[str, torch.Tensor] = {}
    for k in base_state.keys():
        if "batches_tracked" in k:
            continue
        w_avg[k] = torch.zeros_like(base_state[k])
        w_sq_avg[k] = torch.zeros_like(base_state[k])
        w_norm[k] = torch.tensor(0.0)

    n_teachers = float(len(teacher_states))
    for k in w_avg.keys():
        for st in teacher_states:
            grad = st[k] - base_state[k]
            norm = torch.norm(grad, p=2)
            if not torch.isfinite(norm):
                continue
            denom = float(norm) if float(norm) > eps else eps
            grad = grad / denom
            w_avg[k] += grad
            w_sq_avg[k] += grad**2
            w_norm[k] += norm
        w_avg[k] = w_avg[k] / n_teachers
        w_sq_avg[k] = w_sq_avg[k] / n_teachers
        w_norm[k] = w_norm[k] / n_teachers
    return w_avg, w_sq_avg, w_norm


def _sample_swag_gaussian(
    *,
    base_state: StateDict,
    teacher_states: Sequence[StateDict],
    num_samples: int,
    var_scale: float,
    swag_stepsize: float,
    concentrate_num: int = 1,
    var_clamp: float = 1e-6,
) -> List[StateDict]:
    if num_samples <= 0:
        return []
    w_avg, w_sq_avg, w_norm = _compute_mean_sq(base_state=base_state, teacher_states=teacher_states)
    w_var: StateDict = {}
    for k in w_avg.keys():
        w_var[k] = torch.clamp(w_sq_avg[k] - w_avg[k] ** 2, min=var_clamp)

    out: List[StateDict] = []
    for _ in range(int(num_samples)):
        sample_state: StateDict = {}
        for k in w_avg.keys():
            mean_grad = None
            for i in range(max(1, int(concentrate_num))):
                eps = torch.randn_like(w_avg[k])
                sample_grad = w_avg[k] + torch.sqrt(w_var[k]) * eps * float(var_scale)
                if mean_grad is None:
                    mean_grad = sample_grad
                else:
                    mean_grad = (i * mean_grad + sample_grad) / float(i + 1)
            if mean_grad is None:
                mean_grad = torch.zeros_like(w_avg[k])
            sample_state[k] = mean_grad * float(swag_stepsize) * w_norm[k] + base_state[k]
        for k in base_state.keys():
            if k not in sample_state:
                sample_state[k] = base_state[k].clone()
        out.append(sample_state)
    return out


def _sample_dirichlet(
    *,
    teacher_states: Sequence[StateDict],
    num_samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[StateDict]:
    if num_samples <= 0:
        return []
    alpha = float(alpha) if float(alpha) > 0 else 1.0
    out: List[StateDict] = []
    for _ in range(int(num_samples)):
        props = rng.dirichlet(np.repeat(alpha, len(teacher_states)))
        sample_state: StateDict = {}
        for k in teacher_states[0].keys():
            acc = torch.zeros_like(teacher_states[0][k])
            for st, p in zip(teacher_states, props):
                acc = acc + st[k] * float(p)
            sample_state[k] = acc
        out.append(sample_state)
    return out


def _sample_random_mean(
    *,
    teacher_states: Sequence[StateDict],
    num_samples: int,
    rng: np.random.Generator,
    k: int = 3,
) -> List[StateDict]:
    if num_samples <= 0:
        return []
    k = max(1, min(int(k), len(teacher_states)))
    out: List[StateDict] = []
    for _ in range(int(num_samples)):
        idx = rng.choice(len(teacher_states), size=k, replace=False)
        sample_state: StateDict = {}
        for name in teacher_states[0].keys():
            acc = torch.zeros_like(teacher_states[0][name])
            for i in idx:
                acc = acc + teacher_states[int(i)][name]
            sample_state[name] = acc / float(k)
        out.append(sample_state)
    return out


def sample_teacher_states(
    *,
    base_state: StateDict,
    teacher_states: Sequence[StateDict],
    num_samples: int,
    mode: str = "gaussian",
    alpha: float = 1.0,
    var_scale: float = 0.1,
    swag_stepsize: float = 1.0,
    concentrate_num: int = 1,
    rng: np.random.Generator | None = None,
) -> List[StateDict]:
    if rng is None:
        rng = np.random.default_rng()
    mode = str(mode or "gaussian").lower()
    if mode == "gaussian":
        return _sample_swag_gaussian(
            base_state=base_state,
            teacher_states=teacher_states,
            num_samples=int(num_samples),
            var_scale=float(var_scale),
            swag_stepsize=float(swag_stepsize),
            concentrate_num=int(concentrate_num),
        )
    if mode in {"dir", "dirichlet"}:
        return _sample_dirichlet(
            teacher_states=teacher_states,
            num_samples=int(num_samples),
            alpha=float(alpha),
            rng=rng,
        )
    if mode in {"random", "rand"}:
        return _sample_random_mean(
            teacher_states=teacher_states,
            num_samples=int(num_samples),
            rng=rng,
            k=3,
        )
    raise ValueError(f"unknown sample mode: {mode}")

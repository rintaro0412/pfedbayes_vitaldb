from __future__ import annotations

from typing import Dict, List

import torch

from bayes_federated.bayes_layers import BayesParams
from bayes_federated.bayes_param import normalize_param_type, param_to_std, rho_from_sigma


def _clone_tensor(t: torch.Tensor, *, device: torch.device, requires_grad: bool) -> torch.Tensor:
    out = t.detach().clone().to(device)
    if requires_grad:
        out = torch.nn.Parameter(out)
    return out


def clone_bayes_dict(
    params: Dict[str, BayesParams],
    *,
    device: torch.device,
    requires_grad: bool,
) -> Dict[str, BayesParams]:
    out: Dict[str, BayesParams] = {}
    for k, p in params.items():
        out[k] = BayesParams(
            weight_mu=_clone_tensor(p.weight_mu, device=device, requires_grad=requires_grad),
            weight_logvar=_clone_tensor(p.weight_logvar, device=device, requires_grad=requires_grad),
            bias_mu=_clone_tensor(p.bias_mu, device=device, requires_grad=requires_grad),
            bias_logvar=_clone_tensor(p.bias_logvar, device=device, requires_grad=requires_grad),
        )
    return out


def detach_bayes_dict(params: Dict[str, BayesParams]) -> Dict[str, BayesParams]:
    out: Dict[str, BayesParams] = {}
    for k, p in params.items():
        out[k] = BayesParams(
            weight_mu=p.weight_mu.detach(),
            weight_logvar=p.weight_logvar.detach(),
            bias_mu=p.bias_mu.detach(),
            bias_logvar=p.bias_logvar.detach(),
        )
    return out


def bayes_dict_to_cpu(params: Dict[str, BayesParams]) -> Dict[str, BayesParams]:
    out: Dict[str, BayesParams] = {}
    for k, p in params.items():
        out[k] = BayesParams(
            weight_mu=p.weight_mu.detach().cpu(),
            weight_logvar=p.weight_logvar.detach().cpu(),
            bias_mu=p.bias_mu.detach().cpu(),
            bias_logvar=p.bias_logvar.detach().cpu(),
        )
    return out


def bayes_param_list(params: Dict[str, BayesParams]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for p in params.values():
        out.extend([p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar])
    return out


def clamp_bayes_logvar_(
    params: Dict[str, BayesParams],
    *,
    logvar_min: float | None,
    logvar_max: float | None,
    param_type: str = "logvar",
) -> None:
    if normalize_param_type(param_type) != "logvar":
        return
    if logvar_min is None and logvar_max is None:
        return
    for p in params.values():
        with torch.no_grad():
            if logvar_min is not None and logvar_max is not None:
                p.weight_logvar.clamp_(min=float(logvar_min), max=float(logvar_max))
                p.bias_logvar.clamp_(min=float(logvar_min), max=float(logvar_max))
            elif logvar_min is not None:
                p.weight_logvar.clamp_(min=float(logvar_min))
                p.bias_logvar.clamp_(min=float(logvar_min))
            elif logvar_max is not None:
                p.weight_logvar.clamp_(max=float(logvar_max))
                p.bias_logvar.clamp_(max=float(logvar_max))


def _normalize_weights(weights: List[float], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    w = torch.tensor(weights, dtype=dtype, device=device)
    if not torch.isfinite(w).all():
        raise ValueError("Non-finite weights.")
    w_sum = float(w.sum().item())
    if w_sum <= 0:
        raise ValueError("Sum of weights must be > 0.")
    return w / w_sum


def _std_to_param(
    std: torch.Tensor,
    *,
    param_type: str,
    logvar_min: float | None,
    logvar_max: float | None,
) -> torch.Tensor:
    std = std.clamp_min(1e-12)
    ptype = normalize_param_type(param_type)
    if ptype == "rho":
        return rho_from_sigma(std)
    logvar = 2.0 * torch.log(std)
    if logvar_min is not None:
        logvar = logvar.clamp(min=float(logvar_min))
    if logvar_max is not None:
        logvar = logvar.clamp(max=float(logvar_max))
    return logvar


def aggregate_bayes_params(
    *,
    prev: BayesParams,
    locals: List[BayesParams],
    server_beta: float,
    weights: List[float],
    logvar_min: float | None,
    logvar_max: float | None,
    param_type: str = "logvar",
) -> BayesParams:
    if not locals:
        raise ValueError("No local params to aggregate.")
    beta = float(server_beta)
    if beta < 0.0 or beta > 1.0:
        raise ValueError(f"server_beta must be in [0, 1], got {beta}")
    w = _normalize_weights(weights, dtype=prev.weight_mu.dtype, device=prev.weight_mu.device)
    ptype = normalize_param_type(param_type)
    weight_mu_stack = torch.stack([lp.weight_mu for lp in locals], dim=0)
    bias_mu_stack = torch.stack([lp.bias_mu for lp in locals], dim=0)
    weight_std_stack = torch.stack(
        [
            param_to_std(lp.weight_logvar, param_type=ptype, logvar_min=logvar_min, logvar_max=logvar_max)
            for lp in locals
        ],
        dim=0,
    )
    bias_std_stack = torch.stack(
        [
            param_to_std(lp.bias_logvar, param_type=ptype, logvar_min=logvar_min, logvar_max=logvar_max)
            for lp in locals
        ],
        dim=0,
    )
    prev_weight_std = param_to_std(prev.weight_logvar, param_type=ptype, logvar_min=logvar_min, logvar_max=logvar_max)
    prev_bias_std = param_to_std(prev.bias_logvar, param_type=ptype, logvar_min=logvar_min, logvar_max=logvar_max)

    w_broadcast_w = w.view(-1, *([1] * (weight_mu_stack.dim() - 1)))
    w_broadcast_b = w.view(-1, *([1] * (bias_mu_stack.dim() - 1)))
    avg_weight_mu = (w_broadcast_w * weight_mu_stack).sum(dim=0)
    avg_weight_std = (w_broadcast_w * weight_std_stack).sum(dim=0)
    avg_bias_mu = (w_broadcast_b * bias_mu_stack).sum(dim=0)
    avg_bias_std = (w_broadcast_b * bias_std_stack).sum(dim=0)

    # Algorithm 1 (pFedBayes): v^{t+1} = (1-β) v^t + β * Avg(v_i^{t+1})
    # For variance-side params we aggregate in sigma-space (std), then map back.
    new_weight_mu = (1.0 - beta) * prev.weight_mu + beta * avg_weight_mu
    new_weight_std = (1.0 - beta) * prev_weight_std + beta * avg_weight_std
    new_bias_mu = (1.0 - beta) * prev.bias_mu + beta * avg_bias_mu
    new_bias_std = (1.0 - beta) * prev_bias_std + beta * avg_bias_std
    new_weight_logvar = _std_to_param(
        new_weight_std,
        param_type=ptype,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    new_bias_logvar = _std_to_param(
        new_bias_std,
        param_type=ptype,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )

    return BayesParams(
        weight_mu=new_weight_mu.detach().clone(),
        weight_logvar=new_weight_logvar.detach().clone(),
        bias_mu=new_bias_mu.detach().clone(),
        bias_logvar=new_bias_logvar.detach().clone(),
    )


def aggregate_bayes_dict(
    *,
    prev: Dict[str, BayesParams],
    locals: List[Dict[str, BayesParams]],
    server_beta: float,
    weights: List[float],
    logvar_min: float | None,
    logvar_max: float | None,
    param_type: str = "logvar",
) -> Dict[str, BayesParams]:
    if not prev:
        raise ValueError("prev params dict is empty.")
    out: Dict[str, BayesParams] = {}
    for k, pv in prev.items():
        local_list = [lp[k] for lp in locals if k in lp]
        if not local_list:
            out[k] = pv
            continue
        out[k] = aggregate_bayes_params(
            prev=pv,
            locals=local_list,
            server_beta=server_beta,
            weights=weights,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
            param_type=param_type,
        )
    return out

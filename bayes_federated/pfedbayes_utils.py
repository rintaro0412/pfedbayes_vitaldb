from __future__ import annotations

from typing import Dict, List

import torch

from bayes_federated.bayes_layers import BayesParams
from bayes_federated.bayes_param import normalize_param_type


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
    w = _normalize_weights(weights, dtype=prev.weight_mu.dtype, device=prev.weight_mu.device)
    weight_mu = sum(wi * lp.weight_mu for wi, lp in zip(w, locals))
    weight_logvar = sum(wi * lp.weight_logvar for wi, lp in zip(w, locals))
    bias_mu = sum(wi * lp.bias_mu for wi, lp in zip(w, locals))
    bias_logvar = sum(wi * lp.bias_logvar for wi, lp in zip(w, locals))

    new_weight_mu = (1.0 - beta) * prev.weight_mu + beta * weight_mu
    new_weight_logvar = (1.0 - beta) * prev.weight_logvar + beta * weight_logvar
    new_bias_mu = (1.0 - beta) * prev.bias_mu + beta * bias_mu
    new_bias_logvar = (1.0 - beta) * prev.bias_logvar + beta * bias_logvar

    if normalize_param_type(param_type) == "logvar" and (logvar_min is not None or logvar_max is not None):
        if logvar_min is not None and logvar_max is not None:
            new_weight_logvar = new_weight_logvar.clamp(min=float(logvar_min), max=float(logvar_max))
            new_bias_logvar = new_bias_logvar.clamp(min=float(logvar_min), max=float(logvar_max))
        elif logvar_min is not None:
            new_weight_logvar = new_weight_logvar.clamp(min=float(logvar_min))
            new_bias_logvar = new_bias_logvar.clamp(min=float(logvar_min))
        elif logvar_max is not None:
            new_weight_logvar = new_weight_logvar.clamp(max=float(logvar_max))
            new_bias_logvar = new_bias_logvar.clamp(max=float(logvar_max))

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

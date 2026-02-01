from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

ParamType = Literal["logvar", "rho"]


def normalize_param_type(param_type: str | None) -> str:
    if param_type is None:
        return "logvar"
    t = str(param_type).lower().strip()
    return "rho" if t == "rho" else "logvar"


def _clamp_logvar(logvar: torch.Tensor, *, logvar_min: float | None, logvar_max: float | None) -> torch.Tensor:
    if logvar_min is None and logvar_max is None:
        return logvar
    if logvar_min is None:
        return logvar.clamp(max=float(logvar_max))
    if logvar_max is None:
        return logvar.clamp(min=float(logvar_min))
    return logvar.clamp(min=float(logvar_min), max=float(logvar_max))


def param_to_std(
    param: torch.Tensor,
    *,
    param_type: str,
    logvar_min: float | None = None,
    logvar_max: float | None = None,
) -> torch.Tensor:
    t = normalize_param_type(param_type)
    if t == "rho":
        return F.softplus(param)
    logvar = _clamp_logvar(param, logvar_min=logvar_min, logvar_max=logvar_max)
    return torch.exp(0.5 * logvar)


def param_to_var(
    param: torch.Tensor,
    *,
    param_type: str,
    logvar_min: float | None = None,
    logvar_max: float | None = None,
) -> torch.Tensor:
    t = normalize_param_type(param_type)
    if t == "rho":
        std = F.softplus(param)
        return std * std
    logvar = _clamp_logvar(param, logvar_min=logvar_min, logvar_max=logvar_max)
    return torch.exp(logvar)


def param_to_logvar(
    param: torch.Tensor,
    *,
    param_type: str,
    logvar_min: float | None = None,
    logvar_max: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    t = normalize_param_type(param_type)
    if t == "rho":
        std = F.softplus(param)
        return 2.0 * torch.log(std + float(eps))
    return _clamp_logvar(param, logvar_min=logvar_min, logvar_max=logvar_max)


def rho_from_sigma(sigma: torch.Tensor | float) -> torch.Tensor:
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(float(sigma))
    return torch.log(torch.expm1(sigma))

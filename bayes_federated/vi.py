from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from bayes_federated.bayes_layers import BayesParams
from bayes_federated.bayes_param import normalize_param_type, param_to_logvar, param_to_var


@dataclass(frozen=True)
class BetaConfig:
    mode: str = "dataset_scale"  # dataset_scale | fixed
    anneal_steps: int = 0
    base_beta: float = 1.0


@dataclass(frozen=True)
class LossConfig:
    loss_type: str = "bce"  # bce | weighted_bce
    pos_weight: float | None = None


def compute_beta(
    *,
    step: int,
    dataset_size: int,
    cfg: BetaConfig,
) -> float:
    if cfg.mode == "dataset_scale":
        scale = 1.0 / max(int(dataset_size), 1)
    else:
        scale = float(cfg.base_beta)

    if int(cfg.anneal_steps) > 0:
        anneal = min(float(step) / float(cfg.anneal_steps), 1.0)
    else:
        anneal = 1.0
    return float(scale * anneal)


def elbo_loss(
    logits_mc: torch.Tensor,
    targets: torch.Tensor,
    *,
    kl: torch.Tensor,
    beta: float,
    loss_cfg: LossConfig | None = None,
    nll_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logits_mc: (MC, B, 1)
    targets: (B, 1)
    """
    if logits_mc.dim() != 3:
        raise ValueError(f"logits_mc must be (MC,B,1), got {tuple(logits_mc.shape)}")

    if loss_cfg is None:
        loss_cfg = LossConfig()

    targets_mc = targets.unsqueeze(0).expand_as(logits_mc)

    # Flatten MC and batch to reuse existing losses
    logits_flat = logits_mc.reshape(-1, 1)
    targets_flat = targets_mc.reshape(-1, 1)

    pos_weight = None
    if loss_cfg.pos_weight is not None:
        pos_weight = torch.tensor([float(loss_cfg.pos_weight)], device=logits_flat.device, dtype=logits_flat.dtype)

    loss_type = str(loss_cfg.loss_type)
    if loss_type == "weighted_bce":
        nll = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction="mean", pos_weight=pos_weight)
    else:
        nll = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction="mean")

    loss = nll * float(nll_scale) + float(beta) * kl
    return loss, nll.detach(), kl.detach()


def kl_gaussian_params(
    *,
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    logvar_min: float | None = None,
    logvar_max: float | None = None,
    param_type: str = "logvar",
) -> torch.Tensor:
    """
    KL[q || p] for diagonal Gaussians parameterized by (mu, logvar).
    Returns a scalar (sum over all dims). Gradients flow to both q and p.
    """
    ptype = normalize_param_type(param_type)
    logvar_q = param_to_logvar(q_logvar, param_type=ptype, logvar_min=logvar_min, logvar_max=logvar_max)
    logvar_p = param_to_logvar(p_logvar, param_type=ptype, logvar_min=logvar_min, logvar_max=logvar_max)

    var_q = param_to_var(q_logvar, param_type=ptype, logvar_min=logvar_min, logvar_max=logvar_max)
    var_p = param_to_var(p_logvar, param_type=ptype, logvar_min=logvar_min, logvar_max=logvar_max)
    return 0.5 * torch.sum(logvar_p - logvar_q + (var_q + (q_mu - p_mu) ** 2) / var_p - 1.0)


def kl_bayes_params(
    q: BayesParams,
    p: BayesParams,
    *,
    logvar_min: float | None = None,
    logvar_max: float | None = None,
    param_type: str = "logvar",
) -> torch.Tensor:
    return kl_gaussian_params(
        q_mu=q.weight_mu,
        q_logvar=q.weight_logvar,
        p_mu=p.weight_mu,
        p_logvar=p.weight_logvar,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
        param_type=param_type,
    ) + kl_gaussian_params(
        q_mu=q.bias_mu,
        q_logvar=q.bias_logvar,
        p_mu=p.bias_mu,
        p_logvar=p.bias_logvar,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
        param_type=param_type,
    )


def kl_bayes_dict(
    q: dict[str, BayesParams],
    p: dict[str, BayesParams],
    *,
    logvar_min: float | None = None,
    logvar_max: float | None = None,
    param_type: str = "logvar",
) -> torch.Tensor:
    if not q:
        raise ValueError("q params dict is empty")
    if not p:
        raise ValueError("p params dict is empty")
    total = None
    for k, qv in q.items():
        pv = p.get(k)
        if pv is None:
            raise KeyError(f"Missing key in p params: {k}")
        term = kl_bayes_params(qv, pv, logvar_min=logvar_min, logvar_max=logvar_max, param_type=param_type)
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0, device=next(iter(q.values())).weight_mu.device)
    return total

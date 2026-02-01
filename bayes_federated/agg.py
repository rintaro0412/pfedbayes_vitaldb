from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from bayes_federated.bayes_layers import BayesParams


@dataclass(frozen=True)
class AggConfig:
    precision_clamp: float = 1e-6
    # How to form beta_k in prior-corrected PoE (BCM/gPoE).
    # fixed: beta_k = beta_value
    # normalized: beta_k ∝ weights, sum(beta)=1 (gPoE)
    # num_clients: sum(beta)=K (BCM-style evidence accumulation)
    # raw: beta_k = weights (un-normalized)
    beta_mode: str = "normalized"
    beta_value: float = 1.0


@dataclass(frozen=True)
class AggStats:
    clamped_raw: int
    clamped_global: int


def _safe_prec(logvar: torch.Tensor, *, clamp_min: float) -> torch.Tensor:
    var = torch.exp(logvar)
    prec = 1.0 / var
    return prec.clamp(min=float(clamp_min))


def _make_betas(weights: List[float], *, cfg: AggConfig, num_clients: int) -> torch.Tensor:
    w = torch.tensor(weights, dtype=torch.float64)
    if not torch.isfinite(w).all():
        raise ValueError("Non-finite client weights.")
    w_sum = float(w.sum().item())
    if w_sum <= 0:
        raise ValueError("Sum of client weights must be >0.")

    mode = str(cfg.beta_mode).lower()
    if mode == "fixed":
        beta = torch.full_like(w, float(cfg.beta_value))
    elif mode in ("normalized", "mean", "avg"):
        beta = w / w_sum
    elif mode in ("num_clients", "clients", "k"):
        beta = w / w_sum * float(num_clients)
    elif mode in ("raw", "none", "sum"):
        beta = w
    else:
        raise ValueError(f"Unknown agg.beta_mode: {cfg.beta_mode}")
    if not torch.isfinite(beta).all():
        raise ValueError("Non-finite beta values.")
    return beta


def poe_prior_corrected(
    *,
    prior: BayesParams,
    posteriors: List[BayesParams],
    weights: List[float],
    cfg: AggConfig,
) -> tuple[BayesParams, AggStats]:
    if len(posteriors) == 0:
        raise ValueError("No client posteriors to aggregate.")
    if len(posteriors) != len(weights):
        raise ValueError("posteriors and weights length mismatch.")

    beta = _make_betas(weights, cfg=cfg, num_clients=len(posteriors))
    beta_sum = float(beta.sum().item())

    tau_prior_w = _safe_prec(prior.weight_logvar, clamp_min=cfg.precision_clamp)
    tau_prior_b = _safe_prec(prior.bias_logvar, clamp_min=cfg.precision_clamp)
    eta_prior_w = tau_prior_w * prior.weight_mu
    eta_prior_b = tau_prior_b * prior.bias_mu

    tau_w_list = []
    tau_b_list = []
    eta_w_list = []
    eta_b_list = []
    for p in posteriors:
        tau_w = _safe_prec(p.weight_logvar, clamp_min=cfg.precision_clamp)
        tau_b = _safe_prec(p.bias_logvar, clamp_min=cfg.precision_clamp)
        eta_w = tau_w * p.weight_mu
        eta_b = tau_b * p.bias_mu
        tau_w_list.append(tau_w)
        tau_b_list.append(tau_b)
        eta_w_list.append(eta_w)
        eta_b_list.append(eta_b)

    tau_raw_w = sum(bk * tw for bk, tw in zip(beta, tau_w_list)) + (1.0 - beta_sum) * tau_prior_w
    tau_raw_b = sum(bk * tb for bk, tb in zip(beta, tau_b_list)) + (1.0 - beta_sum) * tau_prior_b
    eta_raw_w = sum(bk * ew for bk, ew in zip(beta, eta_w_list)) + (1.0 - beta_sum) * eta_prior_w
    eta_raw_b = sum(bk * eb for bk, eb in zip(beta, eta_b_list)) + (1.0 - beta_sum) * eta_prior_b

    clamped_raw = 0
    if torch.any(tau_raw_w < cfg.precision_clamp):
        clamped_raw += 1
    if torch.any(tau_raw_b < cfg.precision_clamp):
        clamped_raw += 1
    tau_raw_w = tau_raw_w.clamp(min=cfg.precision_clamp)
    tau_raw_b = tau_raw_b.clamp(min=cfg.precision_clamp)

    tau_global_w = tau_raw_w
    tau_global_b = tau_raw_b
    eta_global_w = eta_raw_w
    eta_global_b = eta_raw_b

    clamped_global = 0
    if torch.any(tau_global_w < cfg.precision_clamp):
        clamped_global += 1
    if torch.any(tau_global_b < cfg.precision_clamp):
        clamped_global += 1
    tau_global_w = tau_global_w.clamp(min=cfg.precision_clamp)
    tau_global_b = tau_global_b.clamp(min=cfg.precision_clamp)

    mu_global_w = eta_global_w / tau_global_w
    mu_global_b = eta_global_b / tau_global_b
    logvar_global_w = torch.log(1.0 / tau_global_w)
    logvar_global_b = torch.log(1.0 / tau_global_b)

    if not torch.isfinite(mu_global_w).all() or not torch.isfinite(mu_global_b).all():
        raise ValueError("Non-finite global mean after aggregation.")
    if not torch.isfinite(logvar_global_w).all() or not torch.isfinite(logvar_global_b).all():
        raise ValueError("Non-finite global logvar after aggregation.")

    params = BayesParams(
        weight_mu=mu_global_w.detach().clone(),
        weight_logvar=logvar_global_w.detach().clone(),
        bias_mu=mu_global_b.detach().clone(),
        bias_logvar=logvar_global_b.detach().clone(),
    )
    stats = AggStats(clamped_raw=int(clamped_raw), clamped_global=int(clamped_global))
    return params, stats


def poe_prior_corrected_dict(
    *,
    prior: Dict[str, BayesParams],
    posteriors: List[Dict[str, BayesParams]],
    weights: List[float],
    cfg: AggConfig,
) -> tuple[Dict[str, BayesParams], AggStats]:
    if not prior:
        raise ValueError("prior must be non-empty")
    keys = sorted(prior.keys())
    out: Dict[str, BayesParams] = {}
    clamped_raw = 0
    clamped_global = 0
    for k in keys:
        p0 = prior.get(k)
        if p0 is None:
            continue
        posts = [p[k] for p in posteriors if k in p]
        if not posts:
            out[k] = p0
            continue
        params, stats = poe_prior_corrected(prior=p0, posteriors=posts, weights=weights, cfg=cfg)
        out[k] = params
        clamped_raw += int(stats.clamped_raw)
        clamped_global += int(stats.clamped_global)
    return out, AggStats(clamped_raw=int(clamped_raw), clamped_global=int(clamped_global))


def feduab_aggregate(
    *,
    posteriors: List[BayesParams],
    weights: List[float],
    cfg: AggConfig,
) -> tuple[BayesParams, AggStats]:
    if len(posteriors) == 0:
        raise ValueError("No client posteriors to aggregate.")
    if len(posteriors) != len(weights):
        raise ValueError("posteriors and weights length mismatch.")

    beta = _make_betas(weights, cfg=cfg, num_clients=len(posteriors))

    clamped = 0
    tau_w_list = []
    tau_b_list = []
    mu_w_list = []
    mu_b_list = []
    for p in posteriors:
        tau_w = _safe_prec(p.weight_logvar, clamp_min=cfg.precision_clamp)
        tau_b = _safe_prec(p.bias_logvar, clamp_min=cfg.precision_clamp)
        if torch.any(tau_w < cfg.precision_clamp):
            clamped += 1
        if torch.any(tau_b < cfg.precision_clamp):
            clamped += 1
        tau_w_list.append(tau_w)
        tau_b_list.append(tau_b)
        mu_w_list.append(p.weight_mu)
        mu_b_list.append(p.bias_mu)

    tau_global_w = sum(bk * tw for bk, tw in zip(beta, tau_w_list))
    tau_global_b = sum(bk * tb for bk, tb in zip(beta, tau_b_list))
    tau_global_w = tau_global_w.clamp(min=cfg.precision_clamp)
    tau_global_b = tau_global_b.clamp(min=cfg.precision_clamp)

    eta_global_w = sum(bk * (mw * tw) for bk, mw, tw in zip(beta, mu_w_list, tau_w_list))
    eta_global_b = sum(bk * (mb * tb) for bk, mb, tb in zip(beta, mu_b_list, tau_b_list))

    mu_global_w = eta_global_w / tau_global_w
    mu_global_b = eta_global_b / tau_global_b
    logvar_global_w = torch.log(1.0 / tau_global_w)
    logvar_global_b = torch.log(1.0 / tau_global_b)

    if not torch.isfinite(mu_global_w).all() or not torch.isfinite(mu_global_b).all():
        raise ValueError("Non-finite global mean after aggregation.")
    if not torch.isfinite(logvar_global_w).all() or not torch.isfinite(logvar_global_b).all():
        raise ValueError("Non-finite global logvar after aggregation.")

    params = BayesParams(
        weight_mu=mu_global_w.detach().clone(),
        weight_logvar=logvar_global_w.detach().clone(),
        bias_mu=mu_global_b.detach().clone(),
        bias_logvar=logvar_global_b.detach().clone(),
    )
    stats = AggStats(clamped_raw=int(clamped), clamped_global=int(clamped))
    return params, stats


def feduab_aggregate_dict(
    *,
    posteriors: List[Dict[str, BayesParams]],
    weights: List[float],
    cfg: AggConfig,
) -> tuple[Dict[str, BayesParams], AggStats]:
    if len(posteriors) == 0:
        raise ValueError("No client posteriors to aggregate.")
    keys = sorted(posteriors[0].keys())
    out: Dict[str, BayesParams] = {}
    clamped_raw = 0
    clamped_global = 0
    for k in keys:
        posts = [p[k] for p in posteriors if k in p]
        if not posts:
            continue
        params, stats = feduab_aggregate(posteriors=posts, weights=weights, cfg=cfg)
        out[k] = params
        clamped_raw += int(stats.clamped_raw)
        clamped_global += int(stats.clamped_global)
    return out, AggStats(clamped_raw=int(clamped_raw), clamped_global=int(clamped_global))

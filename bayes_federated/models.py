from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from bayes_federated.bayes_layers import BayesianConv1d, BayesianLinear, BayesianGRU, BayesParams
from bayes_federated.bayes_param import normalize_param_type
from common.ioh_model import ConvBlock1d, IOHModelConfig


@dataclass(frozen=True)
class PointInit:
    weight: torch.Tensor
    bias: torch.Tensor


class BFLModel(nn.Module):
    """
    IOH backbone + Bayesian head.
    """

    def __init__(
        self,
        cfg: IOHModelConfig,
        *,
        prior_sigma: float = 0.1,
        logvar_min: float = -12.0,
        logvar_max: float = 6.0,
        full_bayes: bool = False,
        param_type: str = "logvar",
        mu_init: str = "zeros",
        init_rho: float | None = None,
        var_reduction_h: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.full_bayes = bool(full_bayes)
        self.var_reduction_h = float(var_reduction_h)
        if self.var_reduction_h <= 0:
            raise ValueError("var_reduction_h must be > 0")
        c0 = int(cfg.base_channels)

        layer_idx = 0

        def next_sigma() -> float:
            nonlocal layer_idx
            sigma = float(prior_sigma)
            if self.var_reduction_h != 1.0:
                var = sigma * sigma
                var = var / (self.var_reduction_h ** layer_idx)
                sigma = float(max(var, 1e-12) ** 0.5)
            layer_idx += 1
            return sigma

        if self.full_bayes:
            self.block1 = BayesianConvBlock1d(
                cfg.in_channels,
                c0,
                k=7,
                dropout=cfg.dropout,
                prior_sigma=next_sigma(),
                logvar_min=logvar_min,
                logvar_max=logvar_max,
                param_type=param_type,
                mu_init=mu_init,
                init_rho=init_rho,
            )
            self.block2 = BayesianConvBlock1d(
                c0,
                c0 * 2,
                k=5,
                dropout=cfg.dropout,
                prior_sigma=next_sigma(),
                logvar_min=logvar_min,
                logvar_max=logvar_max,
                param_type=param_type,
                mu_init=mu_init,
                init_rho=init_rho,
            )
            self.block3 = BayesianConvBlock1d(
                c0 * 2,
                c0 * 4,
                k=3,
                dropout=cfg.dropout,
                prior_sigma=next_sigma(),
                logvar_min=logvar_min,
                logvar_max=logvar_max,
                param_type=param_type,
                mu_init=mu_init,
                init_rho=init_rho,
            )
        else:
            self.block1 = ConvBlock1d(cfg.in_channels, c0, k=7, dropout=cfg.dropout)
            self.block2 = ConvBlock1d(c0, c0 * 2, k=5, dropout=cfg.dropout)
            self.block3 = ConvBlock1d(c0 * 2, c0 * 4, k=3, dropout=cfg.dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.use_gru = bool(cfg.use_gru)
        if self.use_gru:
            if self.full_bayes:
                self.gru = BayesianGRU(
                    input_size=c0 * 4,
                    hidden_size=int(cfg.gru_hidden),
                    prior_sigma=next_sigma(),
                    logvar_min=float(logvar_min),
                    logvar_max=float(logvar_max),
                    param_type=param_type,
                    mu_init=mu_init,
                    init_rho=init_rho,
                )
            else:
                self.gru = nn.GRU(
                    input_size=c0 * 4,
                    hidden_size=int(cfg.gru_hidden),
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                )
            head_in = int(cfg.gru_hidden)
        else:
            head_in = c0 * 4

        self.use_clin = int(cfg.clin_dim) > 0
        if self.use_clin:
            if self.full_bayes:
                self.clin_fc = BayesianLinear(
                    int(cfg.clin_dim),
                    int(cfg.clin_dim),
                    prior_sigma=next_sigma(),
                    logvar_min=float(logvar_min),
                    logvar_max=float(logvar_max),
                    param_type=param_type,
                    mu_init=mu_init,
                    init_rho=init_rho,
                )
            else:
                self.clin_fc = nn.Linear(int(cfg.clin_dim), int(cfg.clin_dim))
            self.clin_act = nn.ReLU(inplace=True)
            self.clin_drop = nn.Dropout(p=float(cfg.dropout))
            head_in = int(head_in) + int(cfg.clin_dim)
        else:
            self.clin_fc = None
            self.clin_act = None
            self.clin_drop = None

        if self.full_bayes:
            self.head_fc1 = BayesianLinear(
                head_in,
                head_in,
                prior_sigma=next_sigma(),
                logvar_min=float(logvar_min),
                logvar_max=float(logvar_max),
                param_type=param_type,
                mu_init=mu_init,
                init_rho=init_rho,
            )
        else:
            self.head_fc1 = nn.Linear(head_in, head_in)
        self.head_act = nn.ReLU(inplace=True)
        self.head_drop = nn.Dropout(p=float(cfg.dropout))

        self.bayes_head = BayesianLinear(
            head_in,
            1,
            prior_sigma=next_sigma(),
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
            param_type=param_type,
            mu_init=mu_init,
            init_rho=init_rho,
        )

    def extract_features(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor], *, sample: bool = True) -> torch.Tensor:
        x_clin = None
        if isinstance(x, (tuple, list)):
            if len(x) == 2:
                x, x_clin = x
        if self.full_bayes:
            x = self.block1(x, sample=sample)
        else:
            x = self.block1(x)
        x = self.pool(x)
        if self.full_bayes:
            x = self.block2(x, sample=sample)
        else:
            x = self.block2(x)
        x = self.pool(x)
        if self.full_bayes:
            x = self.block3(x, sample=sample)
        else:
            x = self.block3(x)
        x = self.pool(x)
        if self.use_gru:
            x = x.transpose(1, 2)
            if self.full_bayes:
                feat = self.gru(x, sample=sample)
            else:
                out, _ = self.gru(x)
                feat = out[:, -1, :]
        else:
            feat = x.mean(dim=-1)

        if self.use_clin:
            if x_clin is None:
                raise ValueError("x_clin is required when clin_dim > 0")
            if self.full_bayes:
                clin = self.clin_fc(x_clin, sample=sample)
            else:
                clin = self.clin_fc(x_clin)
            clin = self.clin_act(clin)
            clin = self.clin_drop(clin)
            feat = torch.cat([feat, clin], dim=1)
        if self.full_bayes:
            feat = self.head_fc1(feat, sample=sample)
        else:
            feat = self.head_fc1(feat)
        feat = self.head_act(feat)
        feat = self.head_drop(feat)
        return feat

    def forward(self, x: torch.Tensor, *, sample: bool = True, n_samples: int = 1) -> torch.Tensor:
        if int(n_samples) > 1:
            outs = []
            for _ in range(int(n_samples)):
                feat = self.extract_features(x, sample=True)
                outs.append(self.bayes_head(feat, sample=True))
            return torch.stack(outs, dim=0)
        feat = self.extract_features(x, sample=bool(sample))
        return self.bayes_head(feat, sample=bool(sample))

    def freeze_backbone(self) -> None:
        if not self.full_bayes:
            for name, p in self.named_parameters():
                if name.startswith("bayes_head"):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            return
        for name, p in self.named_parameters():
            if name.startswith("bayes_head") or name.startswith("head_fc1"):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def set_prior(self, params: Dict[str, BayesParams]) -> None:
        if "bayes_head" in params:
            p = params["bayes_head"]
            self.bayes_head.set_prior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if not self.full_bayes:
            return
        if "head_fc1" in params:
            p = params["head_fc1"]
            self.head_fc1.set_prior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        for name in ["block1", "block2", "block3"]:
            if name in params:
                p = params[name]
                getattr(self, name).conv.set_prior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if self.use_clin and "clin_fc" in params:
            p = params["clin_fc"]
            self.clin_fc.set_prior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if self.use_gru:
            gru_params: Dict[str, BayesParams] = {}
            if "gru_ih" in params:
                gru_params["ih"] = params["gru_ih"]
            if "gru_hh" in params:
                gru_params["hh"] = params["gru_hh"]
            if gru_params:
                self.gru.set_priors(gru_params)

    def set_posterior(self, params: Dict[str, BayesParams]) -> None:
        if "bayes_head" in params:
            p = params["bayes_head"]
            self.bayes_head.set_posterior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if not self.full_bayes:
            return
        if "head_fc1" in params:
            p = params["head_fc1"]
            self.head_fc1.set_posterior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        for name in ["block1", "block2", "block3"]:
            if name in params:
                p = params[name]
                getattr(self, name).conv.set_posterior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if self.use_clin and "clin_fc" in params:
            p = params["clin_fc"]
            self.clin_fc.set_posterior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if self.use_gru:
            gru_params: Dict[str, BayesParams] = {}
            if "gru_ih" in params:
                gru_params["ih"] = params["gru_ih"]
            if "gru_hh" in params:
                gru_params["hh"] = params["gru_hh"]
            if gru_params:
                self.gru.set_posteriors(gru_params)

    def get_posterior(self) -> Dict[str, BayesParams]:
        params: Dict[str, BayesParams] = {"bayes_head": self.bayes_head.get_posterior()}
        if not self.full_bayes:
            return params
        params["head_fc1"] = self.head_fc1.get_posterior()
        params["block1"] = self.block1.conv.get_posterior()
        params["block2"] = self.block2.conv.get_posterior()
        params["block3"] = self.block3.conv.get_posterior()
        if self.use_clin:
            params["clin_fc"] = self.clin_fc.get_posterior()
        if self.use_gru:
            gru_params = self.gru.get_posteriors()
            params["gru_ih"] = gru_params["ih"]
            params["gru_hh"] = gru_params["hh"]
        return params

    def get_posterior_params(self) -> Dict[str, BayesParams]:
        """
        Returns posterior parameters as live tensors (with gradients).
        """
        params: Dict[str, BayesParams] = {
            "bayes_head": BayesParams(
                weight_mu=self.bayes_head.weight_mu,
                weight_logvar=self.bayes_head.weight_logvar,
                bias_mu=self.bayes_head.bias_mu,
                bias_logvar=self.bayes_head.bias_logvar,
            )
        }
        if not self.full_bayes:
            return params
        params["head_fc1"] = BayesParams(
            weight_mu=self.head_fc1.weight_mu,
            weight_logvar=self.head_fc1.weight_logvar,
            bias_mu=self.head_fc1.bias_mu,
            bias_logvar=self.head_fc1.bias_logvar,
        )
        params["block1"] = BayesParams(
            weight_mu=self.block1.conv.weight_mu,
            weight_logvar=self.block1.conv.weight_logvar,
            bias_mu=self.block1.conv.bias_mu,
            bias_logvar=self.block1.conv.bias_logvar,
        )
        params["block2"] = BayesParams(
            weight_mu=self.block2.conv.weight_mu,
            weight_logvar=self.block2.conv.weight_logvar,
            bias_mu=self.block2.conv.bias_mu,
            bias_logvar=self.block2.conv.bias_logvar,
        )
        params["block3"] = BayesParams(
            weight_mu=self.block3.conv.weight_mu,
            weight_logvar=self.block3.conv.weight_logvar,
            bias_mu=self.block3.conv.bias_mu,
            bias_logvar=self.block3.conv.bias_logvar,
        )
        if self.use_clin:
            params["clin_fc"] = BayesParams(
                weight_mu=self.clin_fc.weight_mu,
                weight_logvar=self.clin_fc.weight_logvar,
                bias_mu=self.clin_fc.bias_mu,
                bias_logvar=self.clin_fc.bias_logvar,
            )
        if self.use_gru:
            params["gru_ih"] = BayesParams(
                weight_mu=self.gru.ih.weight_mu,
                weight_logvar=self.gru.ih.weight_logvar,
                bias_mu=self.gru.ih.bias_mu,
                bias_logvar=self.gru.ih.bias_logvar,
            )
            params["gru_hh"] = BayesParams(
                weight_mu=self.gru.hh.weight_mu,
                weight_logvar=self.gru.hh.weight_logvar,
                bias_mu=self.gru.hh.bias_mu,
                bias_logvar=self.gru.hh.bias_logvar,
            )
        return params

    def kl_divergence(self) -> torch.Tensor:
        kl = self.bayes_head.kl_divergence()
        if not self.full_bayes:
            return kl
        kl = kl + self.head_fc1.kl_divergence()
        kl = kl + self.block1.conv.kl_divergence()
        kl = kl + self.block2.conv.kl_divergence()
        kl = kl + self.block3.conv.kl_divergence()
        if self.use_clin:
            kl = kl + self.clin_fc.kl_divergence()
        if self.use_gru:
            kl = kl + self.gru.kl_divergence()
        return kl


class BayesianConvBlock1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        k: int,
        dropout: float,
        prior_sigma: float,
        logvar_min: float,
        logvar_max: float,
        param_type: str = "logvar",
        mu_init: str = "zeros",
        init_rho: float | None = None,
    ) -> None:
        super().__init__()
        pad = k // 2
        self.conv = BayesianConv1d(
            in_ch,
            out_ch,
            kernel_size=int(k),
            padding=int(pad),
            prior_sigma=float(prior_sigma),
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
            param_type=param_type,
            mu_init=mu_init,
            init_rho=init_rho,
        )
        groups = 8 if out_ch >= 8 else 1
        self.norm = nn.GroupNorm(groups, out_ch)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        x = self.conv(x, sample=sample)
        x = self.norm(x)
        x = torch.relu(x)
        x = self.drop(x)
        return x


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _extract_point_head(state_dict: Dict[str, torch.Tensor]) -> PointInit | None:
    w_key = "head.3.weight"
    b_key = "head.3.bias"
    if w_key not in state_dict or b_key not in state_dict:
        return None
    return PointInit(weight=state_dict[w_key].detach().clone(), bias=state_dict[b_key].detach().clone())


def _map_backbone_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped = {}
    for k, v in state_dict.items():
        if k.startswith("head.0."):
            mapped[k.replace("head.0.", "head_fc1.")] = v
        elif k.startswith("head.3."):
            continue
        else:
            mapped[k] = v
    return mapped


def build_bfl_model_from_point_checkpoint(
    ckpt_path: str | Path | None,
    *,
    prior_sigma: float,
    logvar_min: float,
    logvar_max: float,
    full_bayes: bool = False,
    fallback_cfg: IOHModelConfig | None = None,
    param_type: str = "logvar",
    mu_init: str = "zeros",
    init_rho: float | None = None,
    var_reduction_h: float = 1.0,
) -> tuple[BFLModel, Dict[str, BayesParams], bool]:
    """
    Returns (model, prior_params, used_point_init).
    """
    ptype = normalize_param_type(param_type)

    if ckpt_path is None or not Path(ckpt_path).exists():
        if fallback_cfg is None:
            fallback_cfg = IOHModelConfig()
        model = BFLModel(
            fallback_cfg,
            prior_sigma=prior_sigma,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
            full_bayes=bool(full_bayes),
            param_type=ptype,
            mu_init=mu_init,
            init_rho=init_rho,
            var_reduction_h=var_reduction_h,
        )
        prior_params = model.get_posterior()
        return model, prior_params, False

    ckpt = torch.load(ckpt_path, map_location="cpu")
    from common.ioh_model import normalize_model_cfg
    cfg = normalize_model_cfg(ckpt.get("model_cfg", {}))
    model = BFLModel(
        cfg,
        prior_sigma=prior_sigma,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
        full_bayes=bool(full_bayes),
        param_type=ptype,
        mu_init=mu_init,
        init_rho=init_rho,
        var_reduction_h=var_reduction_h,
    )

    state_dict = ckpt["state_dict"]
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")
    point_head = _extract_point_head(state_dict)
    if not model.full_bayes:
        mapped = _map_backbone_state(state_dict)
        model.load_state_dict(mapped, strict=False)
        if point_head is None:
            prior_params = model.get_posterior()
            return model, prior_params, False
        with torch.no_grad():
            w_prior = model.bayes_head.prior_weight_logvar.detach().clone().float()
            b_prior = model.bayes_head.prior_bias_logvar.detach().clone().float()
            model.bayes_head.set_posterior(
                point_head.weight.clone().float(),
                w_prior,
                point_head.bias.clone().float(),
                b_prior,
            )
            model.bayes_head.set_prior(
                point_head.weight.clone().float(),
                w_prior,
                point_head.bias.clone().float(),
                b_prior,
            )
        return model, model.get_posterior(), True

    # Full-Bayes: initialize Bayesian params from point weights
    used_point = False
    def _init_linear(module: BayesianLinear, w_key: str, b_key: str | None) -> None:
        nonlocal used_point
        if w_key not in state_dict:
            return
        w = state_dict[w_key].detach().clone().float()
        if b_key and b_key in state_dict:
            b = state_dict[b_key].detach().clone().float()
        else:
            b = torch.zeros((w.shape[0],), dtype=w.dtype)
        param = module.prior_weight_logvar.detach().clone().float()
        bparam = module.prior_bias_logvar.detach().clone().float()
        module.set_posterior(w, param, b, bparam)
        module.set_prior(w, param, b, bparam)
        used_point = True

    def _init_conv(block: BayesianConvBlock1d, w_key: str) -> None:
        nonlocal used_point
        if w_key not in state_dict:
            return
        w = state_dict[w_key].detach().clone().float()
        b = torch.zeros((w.shape[0],), dtype=w.dtype)
        param = block.conv.prior_weight_logvar.detach().clone().float()
        bparam = block.conv.prior_bias_logvar.detach().clone().float()
        block.conv.set_posterior(w, param, b, bparam)
        block.conv.set_prior(w, param, b, bparam)
        used_point = True

    _init_conv(model.block1, "block1.conv.weight")
    _init_conv(model.block2, "block2.conv.weight")
    _init_conv(model.block3, "block3.conv.weight")

    # Load GroupNorm params if available
    for name in ["block1", "block2", "block3"]:
        w_key = f"{name}.norm.weight"
        b_key = f"{name}.norm.bias"
        if w_key in state_dict and b_key in state_dict:
            getattr(model, name).norm.weight.data.copy_(state_dict[w_key].detach().clone())
            getattr(model, name).norm.bias.data.copy_(state_dict[b_key].detach().clone())

    _init_linear(model.head_fc1, "head.0.weight", "head.0.bias")
    if model.use_clin:
        _init_linear(model.clin_fc, "clin_fc.weight", "clin_fc.bias")
    if point_head is not None:
        _init_linear(model.bayes_head, "head.3.weight", "head.3.bias")
    if model.use_gru:
        _init_linear(model.gru.ih, "gru.weight_ih_l0", "gru.bias_ih_l0")
        _init_linear(model.gru.hh, "gru.weight_hh_l0", "gru.bias_hh_l0")

    return model, model.get_posterior(), used_point

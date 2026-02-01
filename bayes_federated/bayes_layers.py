from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn as nn
from bayes_federated.bayes_param import normalize_param_type, param_to_logvar, param_to_std, param_to_var, rho_from_sigma


def _init_linear_mu_bias(weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is None:
        return
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    nn.init.uniform_(bias, -bound, bound)


def _init_conv_mu_bias(weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is None:
        return
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    nn.init.uniform_(bias, -bound, bound)


@dataclass(frozen=True)
class BayesParams:
    weight_mu: torch.Tensor
    weight_logvar: torch.Tensor
    bias_mu: torch.Tensor
    bias_logvar: torch.Tensor


class BayesianLinear(nn.Module):
    """
    Mean-field Gaussian Bayesian linear layer (q(w), q(b)).

    Stores posterior params (mu, logvar) and prior params as buffers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        prior_mu: torch.Tensor | None = None,
        prior_logvar: torch.Tensor | None = None,
        init_mu: torch.Tensor | None = None,
        init_logvar: torch.Tensor | None = None,
        prior_sigma: float = 0.1,
        logvar_min: float = -12.0,
        logvar_max: float = 6.0,
        param_type: str = "logvar",
        mu_init: str = "zeros",
        init_rho: float | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.param_type = normalize_param_type(param_type)

        w_shape = (self.out_features, self.in_features)
        b_shape = (self.out_features,)

        use_pytorch_init = init_mu is None and str(mu_init).lower() in ("pytorch", "kaiming")
        if init_mu is None:
            init_mu = torch.empty(w_shape) if use_pytorch_init else torch.zeros(w_shape)
        if self.param_type == "rho":
            default_param = float(init_rho) if init_rho is not None else float(rho_from_sigma(prior_sigma))
        else:
            default_param = float(torch.log(torch.tensor(prior_sigma**2)))
        if init_logvar is None:
            init_logvar = torch.full(w_shape, default_param)
        if init_mu.shape != w_shape:
            raise ValueError(f"init_mu shape {init_mu.shape} != {w_shape}")
        if init_logvar.shape != w_shape:
            raise ValueError(f"init_logvar shape {init_logvar.shape} != {w_shape}")

        self.weight_mu = nn.Parameter(init_mu.clone().float())
        self.weight_logvar = nn.Parameter(init_logvar.clone().float())

        init_b_mu = torch.zeros(b_shape)
        init_b_logvar = torch.full(b_shape, default_param)

        self.bias_mu = nn.Parameter(init_b_mu.clone().float())
        self.bias_logvar = nn.Parameter(init_b_logvar.clone().float())

        if use_pytorch_init:
            _init_linear_mu_bias(self.weight_mu, self.bias_mu)

        if prior_mu is None:
            prior_mu = torch.zeros(w_shape)
        if prior_logvar is None:
            prior_logvar = torch.full(w_shape, default_param)
        if prior_mu.shape != w_shape:
            raise ValueError(f"prior_mu shape {prior_mu.shape} != {w_shape}")
        if prior_logvar.shape != w_shape:
            raise ValueError(f"prior_logvar shape {prior_logvar.shape} != {w_shape}")

        self.register_buffer("prior_weight_mu", prior_mu.clone().float())
        self.register_buffer("prior_weight_logvar", prior_logvar.clone().float())
        self.register_buffer("prior_bias_mu", torch.zeros(b_shape))
        self.register_buffer("prior_bias_logvar", torch.full(b_shape, default_param))

    def set_prior(self, mu: torch.Tensor, logvar: torch.Tensor, bias_mu: torch.Tensor, bias_logvar: torch.Tensor) -> None:
        device = self.weight_mu.device
        self.prior_weight_mu = mu.detach().clone().float().to(device)
        self.prior_weight_logvar = logvar.detach().clone().float().to(device)
        self.prior_bias_mu = bias_mu.detach().clone().float().to(device)
        self.prior_bias_logvar = bias_logvar.detach().clone().float().to(device)

    def set_posterior(self, mu: torch.Tensor, logvar: torch.Tensor, bias_mu: torch.Tensor, bias_logvar: torch.Tensor) -> None:
        device = self.weight_mu.device
        with torch.no_grad():
            self.weight_mu.copy_(mu.to(device))
            self.weight_logvar.copy_(logvar.to(device))
            self.bias_mu.copy_(bias_mu.to(device))
            self.bias_logvar.copy_(bias_logvar.to(device))

    def get_posterior(self) -> BayesParams:
        return BayesParams(
            weight_mu=self.weight_mu.detach().clone().cpu(),
            weight_logvar=self.weight_logvar.detach().clone().cpu(),
            bias_mu=self.bias_mu.detach().clone().cpu(),
            bias_logvar=self.bias_logvar.detach().clone().cpu(),
        )

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        if sample:
            std_w = param_to_std(self.weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
            std_b = param_to_std(self.bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
            eps_w = torch.randn_like(std_w)
            eps_b = torch.randn_like(std_b)
            w = self.weight_mu + std_w * eps_w
            b = self.bias_mu + std_b * eps_b
        else:
            w = self.weight_mu
            b = self.bias_mu
        return x.matmul(w.t()) + b

    def kl_divergence(self) -> torch.Tensor:
        logvar_q_w = param_to_logvar(self.weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        logvar_q_b = param_to_logvar(self.bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        logvar_p_w = param_to_logvar(self.prior_weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        logvar_p_b = param_to_logvar(self.prior_bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)

        var_q_w = param_to_var(self.weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        var_p_w = param_to_var(self.prior_weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        var_q_b = param_to_var(self.bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        var_p_b = param_to_var(self.prior_bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)

        kl_w = 0.5 * torch.sum(
            logvar_p_w - logvar_q_w + (var_q_w + (self.weight_mu - self.prior_weight_mu) ** 2) / var_p_w - 1.0
        )
        kl_b = 0.5 * torch.sum(
            logvar_p_b - logvar_q_b + (var_q_b + (self.bias_mu - self.prior_bias_mu) ** 2) / var_p_b - 1.0
        )
        return kl_w + kl_b


class BayesianConv1d(nn.Module):
    """
    Mean-field Gaussian Bayesian Conv1d (q(w), q(b)).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_sigma: float = 0.1,
        logvar_min: float = -12.0,
        logvar_max: float = 6.0,
        param_type: str = "logvar",
        mu_init: str = "zeros",
        init_rho: float | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.param_type = normalize_param_type(param_type)

        w_shape = (self.out_channels, self.in_channels // self.groups, self.kernel_size)
        b_shape = (self.out_channels,)

        if self.param_type == "rho":
            default_param = float(init_rho) if init_rho is not None else float(rho_from_sigma(prior_sigma))
        else:
            default_param = float(torch.log(torch.tensor(prior_sigma**2)))
        use_pytorch_init = str(mu_init).lower() in ("pytorch", "kaiming")
        init_mu = torch.empty(w_shape) if use_pytorch_init else torch.zeros(w_shape)
        init_logvar = torch.full(w_shape, default_param)
        self.weight_mu = nn.Parameter(init_mu.clone().float())
        self.weight_logvar = nn.Parameter(init_logvar.clone().float())

        init_b_mu = torch.zeros(b_shape)
        init_b_logvar = torch.full(b_shape, default_param)
        self.bias_mu = nn.Parameter(init_b_mu.clone().float())
        self.bias_logvar = nn.Parameter(init_b_logvar.clone().float())

        if use_pytorch_init:
            _init_conv_mu_bias(self.weight_mu, self.bias_mu)

        prior_mu = torch.zeros(w_shape)
        prior_logvar = torch.full(w_shape, default_param)
        self.register_buffer("prior_weight_mu", prior_mu.clone().float())
        self.register_buffer("prior_weight_logvar", prior_logvar.clone().float())
        self.register_buffer("prior_bias_mu", torch.zeros(b_shape))
        self.register_buffer("prior_bias_logvar", torch.full(b_shape, default_param))

    def set_prior(self, mu: torch.Tensor, logvar: torch.Tensor, bias_mu: torch.Tensor, bias_logvar: torch.Tensor) -> None:
        device = self.weight_mu.device
        self.prior_weight_mu = mu.detach().clone().float().to(device)
        self.prior_weight_logvar = logvar.detach().clone().float().to(device)
        self.prior_bias_mu = bias_mu.detach().clone().float().to(device)
        self.prior_bias_logvar = bias_logvar.detach().clone().float().to(device)

    def set_posterior(self, mu: torch.Tensor, logvar: torch.Tensor, bias_mu: torch.Tensor, bias_logvar: torch.Tensor) -> None:
        device = self.weight_mu.device
        with torch.no_grad():
            self.weight_mu.copy_(mu.to(device))
            self.weight_logvar.copy_(logvar.to(device))
            self.bias_mu.copy_(bias_mu.to(device))
            self.bias_logvar.copy_(bias_logvar.to(device))

    def get_posterior(self) -> BayesParams:
        return BayesParams(
            weight_mu=self.weight_mu.detach().clone().cpu(),
            weight_logvar=self.weight_logvar.detach().clone().cpu(),
            bias_mu=self.bias_mu.detach().clone().cpu(),
            bias_logvar=self.bias_logvar.detach().clone().cpu(),
        )

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        if sample:
            std_w = param_to_std(self.weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
            std_b = param_to_std(self.bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
            eps_w = torch.randn_like(std_w)
            eps_b = torch.randn_like(std_b)
            w = self.weight_mu + std_w * eps_w
            b = self.bias_mu + std_b * eps_b
        else:
            w = self.weight_mu
            b = self.bias_mu
        return nn.functional.conv1d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def kl_divergence(self) -> torch.Tensor:
        logvar_q_w = param_to_logvar(self.weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        logvar_q_b = param_to_logvar(self.bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        logvar_p_w = param_to_logvar(self.prior_weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        logvar_p_b = param_to_logvar(self.prior_bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)

        var_q_w = param_to_var(self.weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        var_p_w = param_to_var(self.prior_weight_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        var_q_b = param_to_var(self.bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)
        var_p_b = param_to_var(self.prior_bias_logvar, param_type=self.param_type, logvar_min=self.logvar_min, logvar_max=self.logvar_max)

        kl_w = 0.5 * torch.sum(
            logvar_p_w - logvar_q_w + (var_q_w + (self.weight_mu - self.prior_weight_mu) ** 2) / var_p_w - 1.0
        )
        kl_b = 0.5 * torch.sum(
            logvar_p_b - logvar_q_b + (var_q_b + (self.bias_mu - self.prior_bias_mu) ** 2) / var_p_b - 1.0
        )
        return kl_w + kl_b


class BayesianLSTM(nn.Module):
    """
    Single-layer Bayesian LSTM (batch_first=True, unidirectional).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        prior_sigma: float = 0.1,
        logvar_min: float = -12.0,
        logvar_max: float = 6.0,
        param_type: str = "logvar",
        mu_init: str = "zeros",
        init_rho: float | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.ih = BayesianLinear(
            self.input_size,
            4 * self.hidden_size,
            prior_sigma=float(prior_sigma),
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
            param_type=param_type,
            mu_init=mu_init,
            init_rho=init_rho,
        )
        self.hh = BayesianLinear(
            self.hidden_size,
            4 * self.hidden_size,
            prior_sigma=float(prior_sigma),
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
            param_type=param_type,
            mu_init=mu_init,
            init_rho=init_rho,
        )

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        # x: (B, T, input_size)
        if x.ndim != 3:
            raise ValueError(f"BayesianLSTM expects (B,T,C), got {x.shape}")
        bsz, t_steps, _ = x.shape
        h = x.new_zeros((bsz, self.hidden_size))
        c = x.new_zeros((bsz, self.hidden_size))
        for t in range(t_steps):
            gates = self.ih(x[:, t, :], sample=sample) + self.hh(h, sample=sample)
            i, f, g, o = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
        return h

    def kl_divergence(self) -> torch.Tensor:
        return self.ih.kl_divergence() + self.hh.kl_divergence()

    def get_posteriors(self) -> dict[str, BayesParams]:
        return {"ih": self.ih.get_posterior(), "hh": self.hh.get_posterior()}

    def set_priors(self, params: dict[str, BayesParams]) -> None:
        if "ih" in params:
            p = params["ih"]
            self.ih.set_prior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if "hh" in params:
            p = params["hh"]
            self.hh.set_prior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)

    def set_posteriors(self, params: dict[str, BayesParams]) -> None:
        if "ih" in params:
            p = params["ih"]
            self.ih.set_posterior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if "hh" in params:
            p = params["hh"]
            self.hh.set_posterior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)


class BayesianGRU(nn.Module):
    """
    Single-layer Bayesian GRU (batch_first=True, unidirectional).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        prior_sigma: float = 0.1,
        logvar_min: float = -12.0,
        logvar_max: float = 6.0,
        param_type: str = "logvar",
        mu_init: str = "zeros",
        init_rho: float | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.ih = BayesianLinear(
            self.input_size,
            3 * self.hidden_size,
            prior_sigma=float(prior_sigma),
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
            param_type=param_type,
            mu_init=mu_init,
            init_rho=init_rho,
        )
        self.hh = BayesianLinear(
            self.hidden_size,
            3 * self.hidden_size,
            prior_sigma=float(prior_sigma),
            logvar_min=float(logvar_min),
            logvar_max=float(logvar_max),
            param_type=param_type,
            mu_init=mu_init,
            init_rho=init_rho,
        )

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        # x: (B, T, input_size)
        if x.ndim != 3:
            raise ValueError(f"BayesianGRU expects (B,T,C), got {x.shape}")
        bsz, t_steps, _ = x.shape
        h = x.new_zeros((bsz, self.hidden_size))
        for t in range(t_steps):
            gi = self.ih(x[:, t, :], sample=sample)
            gh = self.hh(h, sample=sample)
            i_r, i_z, i_n = gi.chunk(3, dim=1)
            h_r, h_z, h_n = gh.chunk(3, dim=1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1.0 - z) * n + z * h
        return h

    def kl_divergence(self) -> torch.Tensor:
        return self.ih.kl_divergence() + self.hh.kl_divergence()

    def get_posteriors(self) -> dict[str, BayesParams]:
        return {"ih": self.ih.get_posterior(), "hh": self.hh.get_posterior()}

    def set_priors(self, params: dict[str, BayesParams]) -> None:
        if "ih" in params:
            p = params["ih"]
            self.ih.set_prior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if "hh" in params:
            p = params["hh"]
            self.hh.set_prior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)

    def set_posteriors(self, params: dict[str, BayesParams]) -> None:
        if "ih" in params:
            p = params["ih"]
            self.ih.set_posterior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)
        if "hh" in params:
            p = params["hh"]
            self.hh.set_posterior(p.weight_mu, p.weight_logvar, p.bias_mu, p.bias_logvar)

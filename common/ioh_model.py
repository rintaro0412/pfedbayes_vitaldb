from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int, dropout: float):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        # GroupNorm is batch-size agnostic (safer for FL than BatchNorm).
        groups = 8 if out_ch >= 8 else 1
        self.norm = nn.GroupNorm(groups, out_ch)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        return x


@dataclass(frozen=True)
class IOHModelConfig:
    in_channels: int = 4
    base_channels: int = 32
    dropout: float = 0.1
    use_gru: bool = True
    gru_hidden: int = 64
    clin_dim: int = 0


def normalize_model_cfg(raw: Dict[str, Any] | None) -> IOHModelConfig:
    if raw is None:
        return IOHModelConfig()
    cfg = dict(raw)
    if "use_lstm" in cfg and "use_gru" not in cfg:
        cfg["use_gru"] = bool(cfg.pop("use_lstm"))
    if "lstm_hidden" in cfg and "gru_hidden" not in cfg:
        cfg["gru_hidden"] = int(cfg.pop("lstm_hidden"))
    if "rnn_type" in cfg and "use_gru" not in cfg:
        cfg["use_gru"] = str(cfg["rnn_type"]).lower() == "gru"
    allowed = {"in_channels", "base_channels", "dropout", "use_gru", "gru_hidden", "clin_dim"}
    cleaned = {k: v for k, v in cfg.items() if k in allowed}
    return IOHModelConfig(**cleaned)


class IOHNet(nn.Module):
    """
    Minimal 1D CNN (optionally + GRU) for IOH prediction.
    Input: (B, C=4, T=6000)
    Output: logits (B, 1)
    """

    def __init__(self, cfg: IOHModelConfig):
        super().__init__()
        c0 = int(cfg.base_channels)
        self.cfg = cfg

        self.block1 = ConvBlock1d(cfg.in_channels, c0, k=7, dropout=cfg.dropout)
        self.block2 = ConvBlock1d(c0, c0 * 2, k=5, dropout=cfg.dropout)
        self.block3 = ConvBlock1d(c0 * 2, c0 * 4, k=3, dropout=cfg.dropout)

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.use_gru = bool(cfg.use_gru)
        if self.use_gru:
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
            self.clin_fc = nn.Linear(int(cfg.clin_dim), int(cfg.clin_dim))
            self.clin_act = nn.ReLU(inplace=True)
            self.clin_drop = nn.Dropout(p=float(cfg.dropout))
            head_in = int(head_in) + int(cfg.clin_dim)
        else:
            self.clin_fc = None
            self.clin_act = None
            self.clin_drop = None

        self.head = nn.Sequential(
            nn.Linear(head_in, head_in),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(head_in, 1),
        )

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor], x_clin: torch.Tensor | None = None) -> torch.Tensor:
        if x_clin is None and isinstance(x, (tuple, list)):
            if len(x) == 2:
                x, x_clin = x  # type: ignore[misc]
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)

        if self.use_gru:
            # (B, C, T) -> (B, T, C)
            x = x.transpose(1, 2)
            out, _ = self.gru(x)
            feat = out[:, -1, :]
        else:
            feat = x.mean(dim=-1)

        if self.use_clin:
            if x_clin is None:
                raise ValueError("x_clin is required when clin_dim > 0")
            clin = self.clin_fc(x_clin)
            clin = self.clin_act(clin)
            clin = self.clin_drop(clin)
            feat = torch.cat([feat, clin], dim=1)

        logits = self.head(feat)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = str(reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1) or (B,)
        targets: (B, 1) or (B,) float{0,1}
        """
        logits = logits.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = (self.alpha * (1 - p_t) ** self.gamma) * bce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

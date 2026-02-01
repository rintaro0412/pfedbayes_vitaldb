import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight=None, reduction: str = "mean"):
        """
        Numerically stable focal loss built on BCE with logits.
        Args:
            alpha (float): Kept for API compatibility; not explicitly used.
            gamma (float): Focusing parameter that down-weights easy examples.
            pos_weight (Tensor): Optional class weighting passed to BCE.
            reduction (str): One of 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

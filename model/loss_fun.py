import torch.nn as nn
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
        assert gamma >= 0, "gamma must be non-negative"
        assert reduction in (
            "mean",
            "sum",
            "none",
        ), f"reduction must be one of ('mean', 'sum', 'none'), got '{reduction}'"
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_weight * focal_weight * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

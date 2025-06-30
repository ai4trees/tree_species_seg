"""Loss function that combines cross-entropy loss and Dice loss."""

__all__ = ["CombinedLoss"]

from typing import Any, Dict, Optional

import segmentation_models_pytorch as smp
import torch


class CombinedLoss(torch.nn.Module):
    """Loss function that combines cross-entropy loss and Lovasz loss."""

    def __init__(
        self,
        lovasz_weight: float = 0.5,
        ce_weight: float = 0.5,
        ce_kwargs: Optional[Dict[str, Any]] = None,
        lovasz_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.lovasz_loss = smp.losses.LovaszLoss(mode="multiclass", **lovasz_kwargs)
        ce_kwargs = ce_kwargs or {}
        self.ce_loss = torch.nn.CrossEntropyLoss(**ce_kwargs)
        self.lovasz_weight = lovasz_weight
        self.ce_weight = ce_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        lovasz = self.lovasz_loss(outputs, targets)
        ce = self.ce_loss(outputs, targets)
        return self.lovasz_weight * lovasz + self.ce_weight * ce

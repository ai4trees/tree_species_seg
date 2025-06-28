"""Loss function that combines cross-entropy loss and Dice loss."""

__all__ = ["CombinedLoss"]

import segmentation_models_pytorch as smp
import torch


class CombinedLoss(torch.nn.Module):
    """Loss function that combines cross-entropy loss and Dice loss."""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(outputs, targets)
        ce = self.ce_loss(outputs, targets)
        return self.dice_weight * dice + self.ce_weight * ce

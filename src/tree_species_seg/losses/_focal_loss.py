"""Focal loss."""

__all__ = ["FocalLoss"]

from typing import Literal, Optional

import torch
import torch.nn.functional

from ._semantic_segmentation_loss import SemanticSegmentationLoss
from ._cross_entropy_loss import CrossEntropyLoss


class FocalLoss(SemanticSegmentationLoss):

    def __init__(
        self,
        apply_softmax: bool = True,
        clip_value: float = 100.0,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__(apply_softmax=apply_softmax, label_smoothing=label_smoothing, reduction=reduction)

        self.gamma = gamma

        self._cross_entropy_loss = CrossEntropyLoss(
            apply_softmax=apply_softmax,
            clip_value=clip_value,
            label_smoothing=label_smoothing,
            reduction="none",
            weight=weight,
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        cross_entropy_loss = self._cross_entropy_loss(prediction, target)

        probabilities = torch.nn.functional.softmax(prediction, dim=-1) if self.apply_softmax else prediction
        one_hot_target = self.one_hot_encode(target, prediction.size(-1))
        probabilities = (probabilities * one_hot_target).sum(dim=-1)

        focal_loss = (1.0 - probabilities) ** self.gamma * cross_entropy_loss  # shape (B, N, C)

        return self._reduce_loss(focal_loss)

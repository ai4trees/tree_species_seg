"""Base class for implementing semantic segmentation losses."""

__all__ = ["SemanticSegmentationLoss"]

from abc import ABC
from typing import Literal

import torch


class SemanticSegmentationLoss(torch.nn.Module, ABC):

    def __init__(
        self,
        apply_softmax: bool = True,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.apply_softmax = apply_softmax
        self.label_smoothing = label_smoothing
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method.")
        self.reduction = reduction

    def one_hot_encode(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:

        target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes)  # pylint: disable=not-callable

        return target_one_hot

    def smooth_label(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:

        if self.label_smoothing == 0:
            return self.one_hot_encode(target, num_classes).double()
        negative_probability = self.label_smoothing / num_classes
        positive_probability = (1 - self.label_smoothing) + negative_probability
        smoothed_label = torch.full(
            (len(target), num_classes),
            fill_value=negative_probability,
            device=target.device,
            dtype=torch.double,
        )
        smoothed_label = torch.scatter(smoothed_label, -1, target.unsqueeze(-1), positive_probability)
        return smoothed_label

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

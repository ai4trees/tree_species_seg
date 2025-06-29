"""Cross-entropy loss."""

__all__ = ["CrossEntropyLoss"]

from typing import Literal, Optional

import torch

from ._semantic_segmentation_loss import SemanticSegmentationLoss


class CrossEntropyLoss(SemanticSegmentationLoss):

    def __init__(
        self,
        apply_softmax: bool = True,
        clip_value: float = 100.0,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__(apply_softmax=apply_softmax, label_smoothing=label_smoothing, reduction=reduction)

        self.register_buffer("clip_value", torch.tensor(clip_value, dtype=torch.float), persistent=False)
        self.register_buffer("weight", weight, persistent=False)

        if self.apply_softmax:
            self._cross_entropy_loss = torch.nn.CrossEntropyLoss(
                reduction="none", weight=weight, label_smoothing=label_smoothing
            )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.apply_softmax:
            cross_entropy_loss = self._cross_entropy_loss(prediction, target)
        else:
            target = self.smooth_label(target, prediction.size(-1))
            cross_entropy_loss = torch.minimum(-target * torch.log(prediction), self.clip_value)
            cross_entropy_loss[target == 0] = 0
            if self.weight is not None:
                cross_entropy_loss *= self.weight
            cross_entropy_loss = cross_entropy_loss.sum(dim=-1)

        return self._reduce_loss(cross_entropy_loss)

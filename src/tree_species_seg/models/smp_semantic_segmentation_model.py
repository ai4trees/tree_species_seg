"""Lightning Module for Forest Species Semantic Segmentation."""

from typing import Any, Dict, Optional, Tuple, Union, Type, cast

import lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
import segmentation_models_pytorch as smp

from tree_species_seg.losses import CombinedLoss


def initialize_model(config, checkpoint_path=None):
    """Initialize model with or without transfer learning."""

    if not checkpoint_path:
        print("Initializing new model")
        return ForestSemanticSegmentationModule(**config)

    # Load checkpoint to check parameters
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    if checkpoint["hyper_parameters"]["num_classes"] == config["num_classes"]:
        return ForestSemanticSegmentationModule.load_from_checkpoint(checkpoint_path=checkpoint_path, **config)

    # Transfer learning if num_classes differ
    print("Initializing transfer learning.")
    return ForestSemanticSegmentationModule.load_for_transfer_learning(
        checkpoint_path=checkpoint_path,
        new_num_classes=config["num_classes"],
        freeze_encoder=config.get("freeze_encoder", True),
        freeze_decoder_except_final=config.get("freeze_decoder_except_final", True),
        learning_rate=config["learning_rate"] * 0.1,
    )


class ForestSemanticSegmentationModule(pl.LightningModule):
    """Lightning Module for Forest Species Semantic Segmentation."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        num_classes: int,
        in_channels: int = 3,
        learning_rate: float = 1e-3,  # pylint: disable=unused-argument
        weight_decay: float = 1e-5,  # pylint: disable=unused-argument
        model_name: str = "unet",
        encoder_name: str = "resnet34",
        loss_type: Union[str, Type[nn.Module]] = "cross_entropy",
        max_epochs: int = 100,  # pylint: disable=unused-argument
        transfer_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the segmentation module.

        Args:
            num_classes: Number of classes to segment
            in_channels: Number of input channels (3 for RGB, 4 for RGBI)
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            model_name: Name of the segmentation model ('unet', 'unetpp', 'deeplabv3', etc.)
            encoder_name: Name of the encoder backbone
            loss_type: Type of loss function to use
            max_epochs: Maximum number of epochs (needed for cosine scheduler)
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = self._create_model(model_name, encoder_name, in_channels, num_classes)
        self.transfer_config = transfer_config
        self.loss = self._create_loss_function(loss_type)

        # Create global metrics (no per-step logging)
        self._setup_metrics(num_classes)

        # Track loss
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

    def _setup_metrics(self, num_classes):
        """Setup global and per-class metrics."""
        # Global metrics
        global_metrics = torchmetrics.MetricCollection(
            {
                "iou": torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=-1),
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, ignore_index=-1),
            }
        )

        # Per-class metrics
        self.per_class_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, ignore_index=-1, average=None
        )
        self.per_class_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, ignore_index=-1, average=None
        )

        # Clone for train/val/test
        self.train_metrics = global_metrics.clone(prefix="train/")
        self.val_metrics = global_metrics.clone(prefix="val/")
        self.test_metrics = global_metrics.clone(prefix="test/")

    @classmethod
    def load_for_transfer_learning(
        cls,
        checkpoint_path: str,
        new_num_classes: int,
        freeze_encoder: bool = True,
        freeze_decoder_except_final: bool = False,
        learning_rate: float = 1e-4,
    ):
        """Load model checkpoint for transfer learning."""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        # Create new model instance with new number of classes
        model = cls(
            num_classes=new_num_classes,
            in_channels=checkpoint["hyper_parameters"]["in_channels"],
            model_name=checkpoint["hyper_parameters"]["model_name"],
            encoder_name=checkpoint["hyper_parameters"]["encoder_name"],
            loss_type=checkpoint["hyper_parameters"]["loss_type"],
            learning_rate=learning_rate,
            transfer_config={
                "freeze_encoder": freeze_encoder,
                "freeze_decoder_except_final": freeze_decoder_except_final,
            },
        )

        # Load state dict except final layer
        pretrained_dict = checkpoint["state_dict"]
        model_state_dict = model.state_dict()

        # Filter out final layer parameters
        filtered_dict = {k: v for k, v in pretrained_dict.items() if "segmentation_head" not in k}

        # Update only the layers that match
        model_state_dict.update(filtered_dict)
        model.load_state_dict(model_state_dict, strict=False)

        # Apply freezing configuration
        if freeze_encoder:
            for param in model.model.encoder.parameters():
                param.requires_grad = False

        if freeze_decoder_except_final:
            for name, param in model.model.decoder.named_parameters():
                if "segmentation_head" not in name:
                    param.requires_grad = False

        return model

    def _create_model(self, model_name: str, encoder_name: str, in_channels: int, num_classes: int):
        """Create segmentation model from smp library."""

        model_map = {
            "unet": smp.Unet,
            "unetpp": smp.UnetPlusPlus,
            "deeplabv3": smp.DeepLabV3,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "pan": smp.PAN,
            "pspnet": smp.PSPNet,
            "fpn": smp.FPN,
            "segformer": smp.Segformer,
        }

        model_cls = model_map.get(model_name.lower())
        if model_cls is None:
            raise ValueError(f"Unsupported model: {model_name}")

        return model_cls(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    def _create_loss_function(self, loss_type: Union[str, Type[nn.Module]]):
        """Create loss function based on specified type."""
        if isinstance(loss_type, nn.Module):
            return loss_type

        loss_type = cast(str, loss_type)

        loss_map = {
            "cross_entropy": nn.CrossEntropyLoss(ignore_index=-1),
            "dice": smp.losses.DiceLoss(mode="multiclass", ignore_index=-1),
            "focal": smp.losses.FocalLoss(mode="multiclass", ignore_index=-1),
            "lovasz": smp.losses.LovaszLoss(mode="multiclass", ignore_index=-1),
            "tversky": smp.losses.TverskyLoss(mode="multiclass", ignore_index=-1),
            "combined": CombinedLoss(ignore_index=-1),
        }

        loss_fn = loss_map.get(loss_type.lower())
        if loss_fn is None:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return loss_fn

    def configure_optimizers(self):
        if self.transfer_config and (
            self.transfer_config["freeze_encoder"] or self.transfer_config["freeze_decoder_except_final"]
        ):
            params = []
            # Add trainable parameters with higher learning rate
            params.append(
                {
                    "params": [p for n, p in self.named_parameters() if p.requires_grad],
                    "lr": self.hparams.learning_rate,
                }
            )
            # Add frozen parameters with zero learning rate
            params.append(
                {
                    "params": [p for n, p in self.named_parameters() if not p.requires_grad],
                    "lr": 0.0,
                }
            )
            optimizer = AdamW(params, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        return self.model(x)

    def _shared_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch["image"], batch["mask"]

        if y.dtype != torch.long:
            y = y.long()

        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        loss, preds, targets = self._shared_step(batch)

        # Update metrics (but don't log per-step)
        self.train_metrics.update(preds, targets)
        self.train_loss.update(loss)

        # Update per-class metrics
        self.per_class_iou.update(preds, targets)
        self.per_class_f1.update(preds, targets)

        # Log only loss per step for progress bar
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        loss, preds, targets = self._shared_step(batch)

        # Update metrics (but don't log per-step)
        self.val_metrics.update(preds, targets)
        self.val_loss.update(loss)

        # Update per-class metrics
        self.per_class_iou.update(preds, targets)
        self.per_class_f1.update(preds, targets)

        # Log only loss per step for progress bar
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def test_step(  # pylint: disable=arguments-differ
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        loss, preds, targets = self._shared_step(batch)

        # Update metrics (but don't log per-step)
        self.test_metrics.update(preds, targets)
        self.test_loss.update(loss)

        # Update per-class metrics
        self.per_class_iou.update(preds, targets)
        self.per_class_f1.update(preds, targets)

        # Log only loss per step for progress bar
        self.log("test/loss", loss, prog_bar=True)

        return loss

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        x = batch["image"]
        logits = self(x)
        return torch.argmax(logits, dim=1)

    def _log_per_class_metrics(self, prefix):
        """Log per-class metrics with proper naming"""
        num_classes = self.hparams.num_classes

        # Compute per-class metrics
        iou_values = self.per_class_iou.compute()
        f1_values = self.per_class_f1.compute()

        # Log each class metric
        for class_idx in range(num_classes):
            self.log(f"{prefix}/class_{class_idx}/iou", iou_values[class_idx], on_epoch=True)
            self.log(f"{prefix}/class_{class_idx}/f1", f1_values[class_idx], on_epoch=True)

        # Reset per-class metrics
        self.per_class_iou.reset()
        self.per_class_f1.reset()

    def on_train_epoch_end(self) -> None:
        # Log global metrics
        self.log("train/loss", self.train_loss.compute(), on_epoch=True)
        self.log_dict(self.train_metrics.compute(), on_epoch=True)

        # Log per-class metrics
        self._log_per_class_metrics("train")

        # Reset metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        # Log global metrics
        self.log("val/loss", self.val_loss.compute(), on_epoch=True)
        self.log_dict(self.val_metrics.compute(), on_epoch=True)

        # Log per-class metrics
        self._log_per_class_metrics("val")

        # Reset metrics
        self.val_loss.reset()
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        # Log global metrics
        self.log("test/loss", self.test_loss.compute(), on_epoch=True)
        self.log_dict(self.test_metrics.compute(), on_epoch=True)

        # Log per-class metrics
        self._log_per_class_metrics("test")

        # Reset metrics
        self.test_loss.reset()
        self.test_metrics.reset()

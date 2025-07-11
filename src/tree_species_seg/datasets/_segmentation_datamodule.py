__all__ = ["SemanticSegmentationDataModule"]

from typing import Any, Dict, List, Optional, Literal, cast

import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler

from ._tree_ai_dataset import TreeAIDataset


class SemanticSegmentationDataModule(pl.LightningDataModule):
    """
    Semantic segmentation data module.

    Args:
        base_dir: Path of the base directory containing the unprocessed dataset.
        output_dir: Path of the directory in which to store the preprocessed dataset.
        batch_size: Batch size for dataloaders.
        num_workers: Number of workers for dataloaders.
        transforms: Optional dict of transform parameters.
        force_reprocess: Whether preprocessing should be repeated if the preprocessed data already exist.
    """

    def __init__(
        self,
        base_dir: str,
        output_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        transforms: Optional[List[str]] = None,
        force_reprocess: bool = False,
        dataset_config: Optional[Dict[str, Any]] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.save_hyperparameters()

        self._base_dir = base_dir
        self._output_dir = output_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._force_reprocess = force_reprocess
        self._dataset_config = dataset_config or {}

        # Will be set in setup()
        self.train_dataset: Optional[TreeAIDataset] = None
        self.val_dataset: Optional[TreeAIDataset] = None
        self.test_dataset: Optional[TreeAIDataset] = None

        # Set up transforms
        self.transforms = transforms or []

    def _get_transforms(self, split: Literal["train", "val", "test"]) -> A.Compose:
        """Get transformations for each split."""

        transforms = []

        transforms_map = {
            "random_brightness_contrast": A.RandomBrightnessContrast(),
            "color_jitter": A.ColorJitter(brightness=(1, 1), hue=(0, 0)),
            "random_gamma": A.RandomGamma(p=0.3),
            "random_shadow": A.RandomShadow(num_shadows_limit=(1, 2), shadow_intensity_range=(0.2, 0.5), p=0.2),
            "random_sun_flare": A.RandomSunFlare(src_radius=64, num_flare_circles_range=(1, 2), p=0.2),
        }

        # Add augmentations for training
        if split == "train":
            transforms.append(
                A.SquareSymmetry(p=0.5),
            )

            for transform in self.transforms:
                if transform not in transforms_map:
                    raise ValueError(f"Invalid transform: {transform}.")
                transforms.append(transforms_map[transform])

        print("Transforms:", transforms)

        # Common transforms for all splits
        transforms.extend(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        return A.Compose(transforms)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each split."""

        if stage == "fit" or stage is None:
            self.train_dataset = TreeAIDataset(
                self._base_dir,
                self._output_dir,
                split="train",
                transforms=self._get_transforms("train"),
                force_reprocess=self._force_reprocess,
                **self._dataset_config,
            )

            self.val_dataset = TreeAIDataset(
                self._base_dir,
                self._output_dir,
                split="val",
                transforms=self._get_transforms("val"),
                force_reprocess=self._force_reprocess,
                **self._dataset_config,
            )
        if stage in ["test", "predict_custom_folder"] or stage is None:
            self.test_dataset = TreeAIDataset(
                self._base_dir,
                self._output_dir,
                split="custom_folder" if stage == "predict_custom_folder" else "test",
                transforms=self._get_transforms("test"),
                force_reprocess=self._force_reprocess,
                **self._dataset_config,
            )

    def train_dataloader(self) -> DataLoader:
        train_dataset = cast(TreeAIDataset, self.train_dataset)
        sampler = None
        if self._dataset_config.get("sampling_weight", "none") != "none":
            print("Using sampling weights", train_dataset.sampling_weights[:10])  # type: ignore[index]
            sampler = WeightedRandomSampler(
                weights=train_dataset.sampling_weights,  # type: ignore[arg-type]
                num_samples=len(train_dataset),
                replacement=True,
            )

        return DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self._num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

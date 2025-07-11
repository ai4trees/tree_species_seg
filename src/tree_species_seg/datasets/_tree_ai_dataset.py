"""TreeAI dataset."""

__all__ = ["TreeAIDataset"]

import json
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import albumentations as A
import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.errors
import torch
from torch.utils.data import Dataset


class TreeAIDataset(Dataset):
    """
    TreeAI dataset.

    Args:
        base_dir: Path of the base directory containing the unprocessed dataset.
        output_dir: Path of the directory in which to store the preprocessed dataset.
        split: Subset of the TreeAI dataset to be used (:code:`"train"` | :code:`"val"` | :code:`"test"`).
        transforms: Optional dict of transform parameters.
        include_partially_labeled_data: Whether the partially labeled data should be included in the dataset.
        ignore_white_and_black_pixels: Whether pixels that are completely white or black should be masked in the loss
            computation.
        force_reprocess: Whether preprocessing should be repeated if the preprocessed data already exist.
    """

    def __init__(
        self,
        base_dir: str,
        output_dir: str,
        *,
        split: Literal["train", "val", "test", "custom_folder"] = "train",
        transforms: Optional[A.Compose] = None,
        include_partially_labeled_data: bool = False,
        ignore_white_and_black_pixels: bool = True,
        force_reprocess: bool = False,
        sampling_weight: Literal["none", "sqrt", "linear"] = "none",
    ):
        super().__init__()
        if split not in ["train", "val", "test", "custom_folder"]:
            raise ValueError(f"Invalid split: {split}.")

        self._base_dir = Path(base_dir)
        self._output_dir = Path(output_dir)
        self._split = split
        self._force_reprocess = force_reprocess
        self.transforms = transforms
        self._include_partially_labeled_data = include_partially_labeled_data
        self._ignore_white_and_black_pixels = ignore_white_and_black_pixels

        self._label_type = "full" if not self._include_partially_labeled_data else "merged"

        if self._ignore_white_and_black_pixels:
            self._label_type = f"{self._label_type}_ignore_white_black"
        else:
            self._label_type = f"{self._label_type}_include_white_black"

        self._dir_partial = None
        self._class_distribution_file = None
        if self._split in ["train", "val"]:
            self._dir_full = self._base_dir / "12_RGB_SemSegm_640_fL" / self._split
            self._dir_partial = self._base_dir / "34_RGB_SemSegm_640_pL" / self._split
            self._class_distribution_file = (
                self._output_dir / self._label_type / self._split / "class_distribution.json"
            )

        elif self._split == "test":
            self._dir_full = self._base_dir / "SemSeg_test-images"
        elif self._split == "custom_folder":
            self._dir_full = self._base_dir

        self._sampling_weight = sampling_weight
        self.sampling_weights: Optional[npt.NDArray] = None

        self.class_mapping = self._get_class_mapping()
        self._img_metadata = self._preprocess_dataset()

    def _get_class_mapping(self) -> Dict[int, int]:
        """
        Returns:
            Dictionary mapping class IDs to consecutive class indices.
        """

        if self._split not in ["train", "val"]:
            return {}

        # read class IDs from stats file
        with open(self._base_dir / "12_RGB_SemSegm_640_fL" / "stats.txt", mode="r", encoding="utf-8") as stats_file:
            lines = stats_file.read().splitlines()
        lines = lines[7:]

        class_mapping = {}
        # the idx 0 indicates background pixels, therefore the species class IDs start with 1
        start_id = 1
        for idx, line in enumerate(lines):
            class_id = int(line.split(" ")[0])
            class_mapping[class_id] = idx + start_id

        return class_mapping

    def _preprocess_test_data(self):
        """
        Prepares the test dataset. This only involves searching the data folder for image files.
        """

        all_images = []
        image_idx = 0
        for file in os.listdir(self._dir_full):
            if Path(file).suffix in (".png", ".tif", ".jpg", ".jpeg"):
                image_path = self._dir_full / file
                if self._split in ["test", "custom_folder"]:
                    metadata = {
                        "idx": image_idx,
                        "image_path": image_path,
                        "label_path": None,
                    }
                    all_images.append(metadata)
                    image_idx += 1

        if len(all_images) == 0:
            raise ValueError("Input data folder is empty.")

        return all_images

    def _preprocess_train_val_data(self):  # pylint: disable=too-many-locals, too-many-branches,too-many-statements
        image_folder = self._dir_full / "images"
        label_folder = self._dir_full / "labels"

        reprocess_dataset = False

        all_images = []
        images = []
        for file in os.listdir(image_folder):
            if file.endswith(".png"):
                images.append((image_folder / file, label_folder / file, True))
        if self._include_partially_labeled_data and self._split in ["train", "val"]:
            partially_labeled_image_folder = self._dir_partial / "images"
            partially_labeled_label_folder = self._dir_partial / "labels"
            for file in os.listdir(partially_labeled_image_folder):
                if file.endswith(".png"):
                    images.append((partially_labeled_image_folder / file, partially_labeled_label_folder / file, False))

        for image_idx, (image_path, label_path, fully_labeled) in enumerate(images):
            file = image_path.name

            image_path_npy = (self._output_dir / self._label_type / self._split / "images" / file).with_suffix(".npy")
            label_path_npy = (self._output_dir / self._label_type / self._split / "labels" / file).with_suffix(".npy")

            if not image_path_npy.exists() or not label_path_npy.exists() or self._force_reprocess:
                reprocess_dataset = True
                try:
                    with rasterio.open(image_path) as img_file:
                        image = img_file.read()
                except rasterio.errors.RasterioIOError:
                    # The training and validation set contain some invalid files that are skipped.
                    continue

                with rasterio.open(label_path) as label_img_file:
                    label_image = label_img_file.read()
                    label_image = np.squeeze(label_image, axis=0)

                label_image = label_image.astype(np.int64)

                if self._ignore_white_and_black_pixels:
                    ignore_mask = np.logical_and(
                        np.logical_or(
                            (image == 0).all(axis=0),
                            (image == 255).all(axis=0),
                        ),
                        label_image == 0,
                    )
                    label_image[ignore_mask] = -1
                if not fully_labeled:
                    label_image[label_image == 0] = -1

                for original_class_idx, remapped_class_idx in self.class_mapping.items():
                    mask = label_image == original_class_idx
                    label_image[mask] = remapped_class_idx

                image_path_npy.parent.mkdir(exist_ok=True, parents=True)
                np.save(image_path_npy, image)

                label_path_npy.parent.mkdir(exist_ok=True, parents=True)
                np.save(label_path_npy, label_image)

            metadata = {
                "idx": image_idx,
                "image_path": image_path_npy,
                "label_path": label_path_npy,
            }
            all_images.append(metadata)
            image_idx += 1

        if reprocess_dataset:
            class_distribution = {class_idx: 0 for class_idx in range(len(self.class_mapping.values()) + 1)}
            for image_info in all_images:
                label_image = np.load(image_info["label_path"])
                class_indices, class_counts = np.unique(label_image, return_counts=True)
                for class_idx, class_count in zip(class_indices, class_counts):
                    if class_idx == -1:
                        continue
                    class_distribution[class_idx] += int(class_count)
            with open(self._class_distribution_file, mode="w", encoding="utf-8") as f:
                json.dump(class_distribution, f)

        if self._split in ["train"] and self._sampling_weight != "none":
            class_distribution_np = np.array(list(class_distribution.values()), dtype=np.uint32)

            if self._sampling_weight == "sqrt":
                class_weights = 1 / np.sqrt(class_distribution_np)
            elif self._sampling_weight == "linear":
                class_weights = 1 / class_distribution_np

            normalization_factor = class_distribution_np.sum() / (class_distribution_np * class_weights).sum()
            class_weights = class_weights * normalization_factor

            self.sampling_weights = np.zeros(len(all_images), dtype=np.float32)

            for idx, image_info in enumerate(all_images):
                label_image = np.load(image_info["label_path"])
                label_image = label_image.flatten()
                label_image = label_image[label_image != -1]

                self.sampling_weights[idx] = class_weights[label_image].sum()
            normalization_factor = len(all_images) / self.sampling_weights.sum()
            self.sampling_weights = self.sampling_weights * normalization_factor
        if self._split == "train" and self._sampling_weight == "none":
            self.sampling_weights = np.ones(len(all_images), dtype=np.uint32)

        return all_images

    def _preprocess_dataset(self):
        """
        Preprocesses the dataset. This involves the following preprocessing steps:

        - Invalid images are removed from the dataset.
        - The images are converted into numpy files.
        - The labels are remapped to consecutive class indices and saved as numpy files.
        """
        print("Preprocess dataset...")

        if self._split in ["train", "val"]:
            return self._preprocess_train_val_data()

        return self._preprocess_test_data()

    def __len__(self) -> int:
        """
        Returns:
            Length of the dataset.
        """

        return len(self._img_metadata)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Returns:
            Dictionary containing the data of the respective image.
        """

        image_info = self._img_metadata[idx]

        if self._split in ["train", "val"]:
            # Load image and mask
            image = np.load(image_info["image_path"])
            mask = np.load(image_info["label_path"])
        else:
            with rasterio.open(image_info["image_path"]) as img_file:
                image = img_file.read()
            mask = None

        # Transpose image to HWC format for albumentations
        image = np.transpose(image, (1, 2, 0))

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        data_item = {
            "id": image_info["image_path"].stem,
            "image": image,
        }
        if mask is not None:
            data_item["mask"] = mask

        return data_item

    def class_distribution(self) -> Optional[Dict[int, int]]:
        if self._class_distribution_file is not None and self._class_distribution_file.exists():
            with open(self._class_distribution_file, mode="r", encoding="utf-8") as f:
                class_distribution = json.load(f)
            return class_distribution

        return None

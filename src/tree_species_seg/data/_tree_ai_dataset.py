"""TreeAI dataset."""

__all__ = ["TreeAIDataset"]

import json
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import albumentations as A
import numpy as np
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
        split: Literal["train", "val", "test"] = "train",
        transforms: Optional[A.Compose] = None,
        include_partially_labeled_data: bool = False,
        ignore_white_and_black_pixels: bool = True,
        force_reprocess: bool = False,
    ):
        super().__init__()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}.")

        self._base_dir = Path(base_dir)
        self._output_dir = Path(output_dir)
        self._split = split
        self._force_reprocess = force_reprocess
        self.transforms = transforms
        self._include_partially_labeled_data = include_partially_labeled_data
        self._ignore_white_and_black_pixels = ignore_white_and_black_pixels

        self._split_dir_partial = None
        if self._split in ["train", "val"]:
            self._split_dir_full = self._base_dir / "12_RGB_SemSegm_640_fL" / self._split
            self._split_dir_partial = self._base_dir / "34_RGB_SemSegm_640_pL" / self._split
        else:
            self._split_dir_full = self._base_dir
        self._label_type = "full" if not self._include_partially_labeled_data else "merged"
        self._class_distribution_file = self._output_dir / self._label_type / self._split / "class_distribution.json"

        self.class_mapping = self._get_class_mapping()
        self.class_distribution = {}
        self._img_metadata = self._preprocess_dataset()

    def _get_class_mapping(self) -> Dict[int, int]:
        """
        Returns:
            Dictionary mapping class IDs to consecutive class indices.
        """

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

    def _preprocess_dataset(self):  # pylint: disable=too-many-locals
        """
        Preprocesses the dataset. This involves the following preprocessing steps:

        - Invalid images are removed from the dataset.
        - The images are converted into numpy files.
        - The labels are remapped to consecutive class indices and saved as numpy files.
        """
        print("Preprocess dataset...")

        image_folder = self._split_dir_full / "images" if self._split in ["train", "val"] else self._split_dir_full
        label_folder = self._split_dir_full / "labels"

        reprocess_dataset = False

        all_images = []
        images = []
        for file in os.listdir(image_folder):
            if file.endswith(".png"):
                images.append((image_folder / file, label_folder / file, True))
        if self._include_partially_labeled_data and self._split in ["train", "val"]:
            partially_labeled_image_folder = self._split_dir_partial / "images"
            partially_labeled_label_folder = self._split_dir_partial / "labels"
            for file in os.listdir(partially_labeled_image_folder):
                if file.endswith(".png"):
                    images.append((partially_labeled_image_folder / file, partially_labeled_label_folder / file, False))

        for image_idx, (image_path, label_path, fully_labeled) in enumerate(images):
            file = image_path.name
            if self._split == "test":
                metadata = {
                    "idx": image_idx,
                    "image_path": image_folder / file,
                    "label_path": None,
                }
                all_images.append(metadata)
                image_idx += 1
            else:
                image_path_npy = (self._output_dir / self._label_type / self._split / "images" / file).with_suffix(
                    ".npy"
                )
                label_path_npy = (self._output_dir / self._label_type / self._split / "labels" / file).with_suffix(
                    ".npy"
                )

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
                        class_distribution[remapped_class_idx] += mask.sum()

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

        if self._split in ["train", "val"] and reprocess_dataset:
            class_distribution = {class_idx: 0 for class_idx in self.class_mapping.values()}
            for image_idx, (image_path, label_path, fully_labeled) in enumerate(images):
                label_path = self._output_dir / self._label_type / self._split / "labels" / label_path.name
                label_image = np.load(label_path)
                class_indices, class_counts = np.unique(label_image, return_counts=True)
                for class_idx, class_count in zip(class_indices, class_counts):
                    class_distribution[class_idx] += class_count
            with open(self._class_distribution_file, mode="w", encoding="utf-8") as f:
                json.dump(class_distribution, f)

        return all_images

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
        if self._class_distribution_file.exists():
            with open(self._class_distribution_file, mode="r", encoding="utf-8") as f:
                class_distribution = json.load(f)
            return class_distribution

        return None

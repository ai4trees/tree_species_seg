"""TreeAI dataset."""

__all__ = ["TreeAIDataset"]

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
        force_reprocess: Whether preprocessing should be repeated if the preprocessed data already exist.
    """

    def __init__(
        self,
        base_dir: str,
        output_dir: str,
        split: Literal["train", "val", "test"] = "train",
        transforms: Optional[A.Compose] = None,
        force_reprocess: bool = False,
    ):
        super.__init__()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}.")

        self._base_dir = Path(base_dir)
        self._output_dir = Path(output_dir)
        self._split = split
        self._force_reprocess = force_reprocess
        self.transforms = transforms

        if self._split in ["train", "val"]:
            self._split_dir = self._base_dir / "12_RGB_SemSegm_640_fL" / self._split
        else:
            self._split_dir = self._base_dir / "SemSeg_test-images"

        self.class_mapping = self._get_class_mapping()
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
        for idx, line in enumerate(lines):
            class_id = int(line.split(" ")[0])
            class_mapping[class_id] = idx

        return class_mapping

    def _preprocess_dataset(self):
        """
        Preprocesses the dataset. This involves the following preprocessing steps:

        - Invalid images are removed from the dataset.
        - The images are converted into numpy files.
        - The labels are remapped to consecutive class indices and saved as numpy files.
        """

        image_folder = self._split_dir / "images" if self._split in ["train", "val"] else self._split_dir
        label_folder = self._split_dir / "labels"

        all_images = []
        image_idx = 0
        for file in os.listdir(image_folder):
            if not file.endswith(".png"):
                continue

            if self._split == "test":
                metadata = {
                    "idx": image_idx,
                    "image_path": image_folder / file,
                    "label_path": None,
                }
                all_images.append(metadata)
                image_idx += 1
            else:
                image_path_npy = (self._output_dir / self._split / "images" / file).with_suffix(".npy")
                label_path_npy = (self._output_dir / self._split / "labels" / file).with_suffix(".npy")

                if not image_path_npy.exists() or not label_path_npy.exists() or self._force_reprocess:
                    try:
                        with rasterio.open(image_folder / file) as img_file:
                            image = img_file.read()
                    except rasterio.errors.RasterioIOError:
                        # The training and validation set contain some invalid files that are skipped.
                        continue

                    with rasterio.open(label_folder / file) as label_img_file:
                        label_image = label_img_file.read()
                        label_image = np.squeeze(label_image, axis=0)

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
        data_item = {}

        if self._split in ["train", "val"]:
            # Load image and mask
            image = np.load(image_info["image_path"])
            mask = np.load(image_info["label_path"])

            # Transpose image to HWC format for albumentations
            image = np.transpose(image, (1, 2, 0))

            # Apply transforms if any
            if self.transforms:
                transformed = self.transforms(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            data_item["image"] = image
            data_item["mask"] = mask
        else:
            with rasterio.open(image_info["image_path"]) as img_file:
                image = img_file.read()

            data_item["image"] = image

        data_item["id"] = image_info["image_path"].stem

        return data_item

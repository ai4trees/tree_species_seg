"""Inference script."""

from importlib_resources import files, as_file
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, cast

import colorcet as cc
import fire
import json
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.enums import ColorInterp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

import tree_species_seg
from tree_species_seg.datasets import SemanticSegmentationDataModule, TreeAIDataset
from tree_species_seg.models.smp_semantic_segmentation_model import ForestSemanticSegmentationModule
import tree_species_seg.pkg_data


def create_visualization(
    image: npt.NDArray,
    color_map: npt.NDArray,
    class_labels: Union[Dict[int, str], List[int], npt.NDArray],
    target: Optional[npt.NDArray] = None,
    prediction: Optional[npt.NDArray] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (25, 7),  # Wider figure to accommodate legend
) -> None:
    """Create and save visualization of prediction results with side legend."""

    fig = plt.figure(figsize=figsize)

    # Columns for images, ground-truth labels, prediction and legend
    columns = 2
    if target is not None:
        columns += 1
    if prediction is not None:
        columns += 1
    gs = fig.add_gridspec(1, columns)

    axes = [fig.add_subplot(gs[0, 0])]
    for column in range(columns - 2):
        axes.append(fig.add_subplot(gs[0, 1 + column]))

    # Original image
    img = image.transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    # Ground truth
    if target is not None:
        target_colored = color_map[target]
        axes[1].imshow(target_colored)
        axes[1].set_title("Ground Truth", fontsize=12)
        axes[1].axis("off")

    # Prediction
    if prediction is not None:
        prediction_colored = color_map[prediction]
        axes[-1].imshow(prediction_colored)
        axes[-1].set_title("Prediction", fontsize=12)
        axes[-1].axis("off")

    class_indices = []
    target_classes = np.unique(target) if target is not None else []
    predicted_classes = np.unique(prediction) if prediction is not None else []
    for i in range(len(class_labels)):
        if i in target_classes or i in predicted_classes:
            class_indices.append(i)

    # Create legend handles
    handles = [plt.Rectangle((0, 0), 1, 1, fc=color_map[i]) for i in class_indices]  # type: ignore[attr-defined]

    legend = fig.legend(
        handles,
        [str(class_labels[i]) for i in class_indices],
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        fontsize=10,
        ncol=1,
    )

    plt.tight_layout()

    if save_path:
        # Save with extra bounding box to include legend
        plt.savefig(save_path, bbox_inches="tight", dpi=300, bbox_extra_artists=[legend])
        plt.close()
    else:
        plt.show()


def inference(
    data_loader: DataLoader,
    class_remapping: Dict[int, int],
    checkpoint: str,
    output_dir: str,
    *,
    output_format: Literal[".npy", ".png"] = ".npy",
    visualization_dir: Optional[str] = None,
):
    """
    Inference script.

    Args:
        data_loader: Data loader for the inference dataset.
        class_remapping: Dictionary defining a mapping between consecutive class IDs and output class IDs.
        checkpoint: Path of the model checkpoint file.
        output_dir: Path of the folder in which to save the predictions.
        output_format: Output file format: :code:`".npy"` | :code:`".png"`
        visualization_dir: Directory in which to save visualizations of the predictions. If set to :code:`None`, no
            visualizations are created.
    """
    if output_format not in [".npy", ".png"]:
        raise ValueError(f"The output format {output_format} is not supported.")

    torch.set_float32_matmul_precision("medium")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    model = ForestSemanticSegmentationModule.load_from_checkpoint(checkpoint)
    model.eval()

    num_classes = len(class_remapping)
    if visualization_dir is not None:
        visualization_dir_path = Path(visualization_dir)
        visualization_dir_path.mkdir(exist_ok=True, parents=True)
        color_map = np.array([mcolors.to_rgb(color) for color in cc.glasbey])
        class_labels = np.arange(num_classes)

    for class_idx in range(num_classes):
        if class_idx not in class_remapping:
            raise ValueError(f"Key {class_idx} is missing in class remapping.")

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Generating predictions"):
            images = batch["image"].to(model.device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            remapped_preds = np.zeros_like(preds)

            for class_idx, class_id in class_remapping.items():
                remapped_preds[preds == class_idx] = class_id
            del preds

            for idx in range(len(batch["id"])):

                prediction_path = output_dir_path / f"{batch['id'][idx]}"

                if output_format == ".npy":
                    prediction_path = prediction_path.with_suffix(".npy")
                    np.save(prediction_path, remapped_preds[idx])
                else:
                    prediction_path = prediction_path.with_suffix(".png")
                    prediction = remapped_preds[idx].astype(np.uint8)
                    with rasterio.open(
                        prediction_path,
                        "w",
                        driver="PNG",
                        height=prediction.shape[0],
                        width=prediction.shape[0],
                        count=1,
                        dtype=np.uint8,
                    ) as output_file:
                        output_file.write(prediction, 1)
                        output_file.colorinterp = [ColorInterp.gray]

                if visualization_dir is not None:
                    visualization_path = visualization_dir_path / f"{batch['id'][idx]}.png"
                    create_visualization(
                        images[idx].cpu().numpy(),
                        color_map,
                        class_labels,
                        prediction=remapped_preds[idx],
                        save_path=visualization_path,
                    )


def inference_tree_ai(
    config: str,
    checkpoint: str,
    output_dir: str,
    *,
    output_format: Literal[".npy", ".png"] = ".npy",
    visualization_dir: Optional[str] = None,
    split: Literal["train", "val", "test"] = "test",
):
    """
    Runs inference on a given split of the TreeAI dataset.

    Args:
        config: Path of the configuration file.
        output_dir: Path of the folder in which to save the predictions.
        output_format: Output file format: :code:`".npy"` | :code:`".png"`
        visualization_dir: Folder in which to save visualizations of the predictions. If set to :code:`None`, no
            visualizations are created.
        split: Subset of the dataset for which to generate predictions: :code:`"train"` | :code:`"val"` | :code:`"test"`.
    """

    with open(config, "r") as file:
        conf = yaml.safe_load(file)

    datamodule = SemanticSegmentationDataModule(**conf["dataset"])

    datamodule.setup(stage="fit" if split in ["train", "val"] else "test")
    if split == "train":
        dataset = datamodule.train_dataset
        data_loader = datamodule.train_dataloader()
    if split == "val":
        dataset = datamodule.val_dataset
        data_loader = datamodule.val_dataloader()
    if split == "test":
        dataset = datamodule.test_dataset
        data_loader = datamodule.test_dataloader()

    dataset = cast(TreeAIDataset, dataset)

    if split in ["train", "val"]:
        class_remapping = {0: 0}

        for class_id, class_idx in dataset.class_mapping.items():
            class_remapping[class_idx] = class_id
    else:
        class_remapping_file = files(tree_species_seg.pkg_data).joinpath("tree_ai_class_remapping.json")
        with as_file(class_remapping_file) as class_remapping_file_path:
            with open(class_remapping_file_path, mode="r", encoding="utf-8") as f:
                class_remapping_json = json.load(f)
                # convert string keys from JSON to integers
                class_remapping = {int(class_idx): class_id for class_idx, class_id in class_remapping_json.items()}

    inference(
        data_loader,
        class_remapping,
        checkpoint,
        output_dir,
        output_format=output_format,
        visualization_dir=visualization_dir,
    )


def inference_folder(
    input_folder: str,
    checkpoint: str,
    output_dir: str,
    output_format: Literal[".npy", ".png"] = ".npy",
    visualization_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 8,
    class_remapping_file: Optional[str] = None,
):
    """
    Runs inference on a folder with image files.

    Args:
        input_folder: Path of the folder containing the input images.
        checkpoint: Path of the model checkpoint file.
        output_dir: Path of the folder in which to save the predictions.
        output_format: Output file format: :code:`".npy"` | :code:`".png"`
        visualization_dir: Folder in which to save visualizations of the predictions. If set to :code:`None`, no
            visualizations are created.
        batch_size: Batch size.
        num_workers: Number of worker processes for dataloading.
        class_remapping_file: Path of a JSON file containing a dictionary with a custom remapping of consecutive class
            indices to class IDs. If set to :code:`None`, the class indices are remapped to the class IDs of the TreeAI
            dataset.
    """

    datamodule = SemanticSegmentationDataModule(
        input_folder, input_folder, batch_size=batch_size, num_workers=num_workers
    )
    datamodule.setup(stage="predict_custom_folder")
    data_loader = datamodule.test_dataloader()

    if class_remapping_file is None:
        class_remapping_file = files(tree_species_seg.pkg_data).joinpath("tree_ai_class_remapping.json")
        with as_file(class_remapping_file) as class_remapping_file_path:
            with open(class_remapping_file_path, mode="r", encoding="utf-8") as f:
                class_remapping_json = json.load(f)
    else:
        class_remapping_file = files(tree_species_seg.pkg_data).joinpath("tree_ai_class_remapping.json")
        with as_file(class_remapping_file) as class_remapping_file_path:
            with open(class_remapping_file_path, mode="r", encoding="utf-8") as f:
                class_remapping_json = json.load(f)
    # convert string keys from JSON to integers
    class_remapping = {int(class_idx): class_id for class_idx, class_id in class_remapping_json.items()}

    inference(
        data_loader,
        class_remapping,
        checkpoint,
        output_dir,
        output_format=output_format,
        visualization_dir=visualization_dir,
    )


if __name__ == "__main__":
    fire.Fire({"tree_ai": inference_tree_ai, "folder": inference_folder})

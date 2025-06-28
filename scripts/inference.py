"""Inference script."""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import yaml

import colorcet as cc
import fire
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from tree_species_seg.data import SemanticSegmentationDataModule
from tree_species_seg.models.smp_semantic_segmentation_model import initialize_model, ForestSemanticSegmentationModule


def create_visualization(
    image: npt.NDArray,
    color_map: npt.NDArray,
    class_labels: Union[Dict[int, str], List[int]],
    target: Optional[npt.NDArray] = None,
    prediction: Optional[npt.NDArray] = None,
    save_path: Path = None,
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
    handles = [plt.Rectangle((0, 0), 1, 1, fc=color_map[i]) for i in class_indices]

    legend = fig.legend(
        handles,
        [class_labels[i] for i in class_indices],
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


def main(
    config: str,
    checkpoint: str,
    output_dir: str,
    visualization_dir: Optional[str] = None,
    split: Literal["train", "val", "test"] = "test",
):
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    torch.set_float32_matmul_precision("medium")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    datamodule = SemanticSegmentationDataModule(**config["dataset"])

    model = ForestSemanticSegmentationModule.load_from_checkpoint(checkpoint)
    model.eval()

    datamodule.setup(stage=split)
    if split == "train":
        data_loader = datamodule.train_dataloader()
    if split == "val":
        data_loader = datamodule.val_dataloader()
    if split == "test":
        data_loader = datamodule.test_dataloader()

    dataset = data_loader.dataset

    if visualization_dir is not None:
        visualization_dir_path = Path(visualization_dir)
        visualization_dir_path.mkdir(exist_ok=True, parents=True)
        num_classes = max(dataset.inverse_class_mapping.values()) + 1
        color_map = np.array([mcolors.to_rgb(color) for color in cc.glasbey])
        class_labels = np.arange(num_classes)

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Generating predictions"):
            images = batch["image"].to(model.device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            remapped_preds = np.zeros_like(preds)

            for class_id, class_idx in dataset.class_mapping.items():
                remapped_preds[preds == class_idx] = class_id
            del preds

            for idx in range(len(batch["id"])):
                prediction_path = output_dir_path / f"{batch['id'][idx]}.npy"
                np.save(prediction_path, remapped_preds[idx])

                if visualization_dir is not None:
                    visualization_path = visualization_dir_path / f"{batch['id'][idx]}.png"
                    create_visualization(
                        images[idx].cpu().numpy(),
                        color_map,
                        class_labels,
                        prediction=remapped_preds[idx],
                        save_path=visualization_path,
                    )


if __name__ == "__main__":
    fire.Fire(main)

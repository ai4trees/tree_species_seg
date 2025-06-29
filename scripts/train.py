"""Training script."""

from pathlib import Path
from typing import Optional

import fire
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
import torch
import yaml

from tree_species_seg.data import SemanticSegmentationDataModule
from tree_species_seg.models.smp_semantic_segmentation_model import initialize_model


def main(
    config: str,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[str] = None,
    checkpoint: Optional[str] = None,
):
    """
    Args:
        config: Path to the configuration file.
        wandb_project: Weights and Biases project name.
        tags: Comma-separated tags for Weights and Biases run.
        checkpoint: Path to checkpoint file to resume from.
    """
    with open(config, "r") as file:
        conf = yaml.safe_load(file)

    # Setup W&B
    wandb_logger = None
    if wandb_project is not None:
        tag_list = tags.split(",") if tags else []
        wandb_run_name = (
            run_name or f"{conf['dataset']['name']}_{conf['model']['model_name']}_{conf['model']['encoder_name']}"
        )
        if checkpoint:
            wandb_run_name += "_resumed"

        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            tags=tag_list,
            log_model=True,  # Logs model checkpoints
            config=conf,  # Logs your configuration
        )

    torch.set_float32_matmul_precision("medium")
    seed_everything(seed=conf["seed"], workers=True)

    datamodule_config = conf["dataset"]
    dataset_config = datamodule_config.get(datamodule_config["name"], {})
    model_config = conf["model"]

    datamodule = SemanticSegmentationDataModule(**datamodule_config, dataset_config=dataset_config)
    model = initialize_model(conf["model"], checkpoint)

    checkpoint_dir_name = f"{model_config['model_name']}_{model_config['encoder_name']}_{model_config['loss_type']}"
    if run_name is not None:
        checkpoint_dir_name = f"{checkpoint_dir_name}_{run_name}"
    checkpoint_dir = Path(conf["training"]["checkpoint_dir"]) / f"{datamodule_config['name']}/{checkpoint_dir_name}"

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="checkpoint-{epoch}-iou={val/iou:.3f}",
            auto_insert_metric_name=False,
            monitor="val/iou",
            mode="max",
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
        ),
        EarlyStopping(monitor="val/iou", mode="max", patience=10, verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=int(model_config["max_epochs"]),
        callbacks=callbacks,
        logger=wandb_logger,
        num_nodes=1,
        devices="auto",
        accelerator="auto",
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        precision="16-mixed",
        accumulate_grad_batches=conf["training"].get("accumulate_grad_batches", 1),
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(main)

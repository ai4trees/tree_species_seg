## Semantic Segmentation of Tree Species in High-Resolution UAV Imagery

Josafat-Mattias Burmeister, David Kuska, Stefan Reder, Rico Richter, Jan-Peter Mund, Jürgen Döllner

### Overview

This repository implements a deep learning pipeline for semantic segmentation of tree species in high-resolution UAV imagery.

### Installation

1. Install a [PyTorch](https://pytorch.org/get-started/locally/) version compatible with your operating system. 
2. Clone our repository and install our Python package:

```bash
git clone ttps://github.com/ai4trees/tree_species_classification_uav.git
cd tree_species_classification_uav
python -m pip install .
```

### Docker Image

We also provide a Docker image that contains our package and its dependencies. You can run a Docker container with the following command:

```
docker run --rm -it -v <path to your data>:/workspace/data josafatburmeister:josafatburmeister/tree_species_seg
```

### Model Training

Models can be trained using the training script. An example command:

```bash
python .\scripts\main_semantic_segmentation.py --config .\config\UNetPlusPlus.yaml --wandb-project tree-ai-seg
```

Example configuration files are included in the `./configs` directory of our repository.

### Inference

To run inference:

```bash
python .\scripts\inference.py  --config .\config\SegFormer.yaml --checkpoint "C:\Users\josafat-1\Documents\Code\models\tree_ai\segformer_resnet50_lovasz-1\epoch=75-val\iou=0.362.ckpt" --output-dir predictions  --visualization-dir visualizations
```

# Cross-Domain Cervical Cancer Analysis Pipeline

This project implements an end-to-end self-supervised pipeline for cervical cancer analysis using dual-domain data (colposcopy and histopathology). The goal is to address the challenge of limited labeled datasets and unpaired modalities.

## Key Features

*   **Dual-Encoder Architecture:** Separate encoders for colposcopy and histopathology images to extract domain-specific features.
*   **Self-Supervised Learning (SSL):** Aligns representations via contrastive learning and prototype distribution matching.
*   **Weakly Supervised Lesion Attention:** Identifies salient regions (lesions) without pixel-level annotations.
*   **Multi-Task Prediction Head:** Estimates malignancy and severity scores simultaneously.
*   **Domain Adaptation:** Transfers knowledge from histopathology to colposcopy without paired samples.

## Project Structure

*   `data/`: Contains dataset loading and preprocessing logic.
*   `models/`: Implements neural network architectures (backbones, attention modules, projection heads).
*   `utils/`: Helper functions for logging, visualization, and metrics.
*   `configs/`: Configuration files (YAML) for hyperparameters.
*   `scripts/`: Training, evaluation, and inference scripts.

## Data Acquisition

For details on how to download and setup external datasets (AnnoCerv, MobileODT, etc.), please see [data_acquisition.md](data_acquisition.md).

## Getting Started

1.  **Installation:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Training:**
    To pre-train the model:
    ```bash
    python scripts/train_ssl.py --config configs/ssl_config.yaml
    ```

    To fine-tune for downstream tasks:
    ```bash
    python scripts/train_finetune.py --config configs/finetune_config.yaml
    ```

3.  **Visualization:**
    See `notebooks/visualization.ipynb` for attention map visualizations.

## Dependencies

*   PyTorch
*   Torchvision
*   NumPy
*   OpenCV
*   Scikit-learn
*   PyYAML
*   tqdm

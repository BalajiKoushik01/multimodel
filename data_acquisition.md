# Data Acquisition & Setup Guide

This project supports several external datasets. Since these datasets often require user agreements or Kaggle API keys, you must download them manually. This guide explains how to get them and where to convert/place them.

## 1. Intel MobileODT Cervical Cancer Screening (Kaggle)

**Source:** [https://www.kaggle.com/competitions/intel-mobileodt-cervical-cancer-screening/data](https://www.kaggle.com/competitions/intel-mobileodt-cervical-cancer-screening/data)

**Steps:**
1.  Download the `train.7z` and `test.7z` files.
2.  Extract them into `data/mobileodt/`.
3.  Structure should look like:
    ```
    data/mobileodt/
      train/
        Type_1/
        Type_2/
        Type_3/
      test/
    ```

**Usage:**
In your config or script, set `dataset_path="data/mobileodt"` and `modality="mobileodt"`.

## 2. AnnoCerv (GitHub)

**Source:** [https://github.com/iclx/AnnoCerv](https://github.com/iclx/AnnoCerv)

**Steps:**
1.  Clone the repository or download the release.
2.  Copy the `Cases` folders (e.g., `Case 1`, `Case 2`, etc.) into `data/annocerv/`.
3.  Structure:
    ```
    data/annocerv/
      Case 1/
        C1Aceto (1).jpg
        C1Aceto (1).png
        ...
      Case 2/
      ...
    ```

**Usage:**
Set `dataset_path="data/annocerv"` and `modality="annocerv"`. The `Dataset` class acts as a wrapper to parse the recursive directory structure.

## 3. Expert Eyes Colposcopy Ratings (Kaggle)

**Source:** [https://www.kaggle.com/datasets/rabieelkharoua/expert-eyes-colposcopy-ratings-dataset](https://www.kaggle.com/datasets/rabieelkharoua/expert-eyes-colposcopy-ratings-dataset)

**Steps:**
1.  Download the dataset.
2.  Place images in `data/expert_eyes/`.

## 4. Other Resources

*   **CerviFusionNet**: [https://github.com/Li-OmicsLab/CerviFusionNet](https://github.com/Li-OmicsLab/CerviFusionNet). (Code only, data upon request).
*   **Roboflow**: [https://blog.roboflow.com/best-image-annotation-tools/](https://blog.roboflow.com/best-image-annotation-tools/). (For annotating your own data).
*   **Colposcopy Assistant**: [https://huggingface.co/spaces/ManalifeAI/Colposcopy_Assistant](https://huggingface.co/spaces/ManalifeAI/Colposcopy_Assistant). (Online demo).

## Data Organization for Training

The `CervicalDataset` class in `data/dataset.py` currently supports:
*   `modality='colposcopy'`: Flat folder of images.
*   `modality='histopathology'`: Flat folder of images.
*   `modality='mobileodt'`: Recurses into `Type_X` folders.
*   `modality='annocerv'`: Recurses into `Case X` folders.

To use these in `train_ssl.py`, update `configs/config.yaml`:
```yaml
dataset_path: "data/mobileodt" # or "data/annocerv"
```
And update `scripts/train_ssl.py` to instantiate the dataset with the correct modality:
```python
dataset_colpo = CervicalDataset(config['dataset_path'], modality='mobileodt', ...)
```

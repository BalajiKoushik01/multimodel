# MultiModel — cervical-imaging research experiments

An exploratory Python machine-learning repository with scripts for data preparation, self-supervised training, fine-tuning, evaluation and visualisation. It is **research software only, not clinically validated** and must not be used for screening, diagnosis or treatment decisions.

## Contents

- `scripts/` — data download, cleaning, synthetic/mock data generation, training, evaluation and inference entry points.
- `configs/config.yaml` — general model configuration including backbone, class assumptions and training settings.
- `utils/` — reusable losses and external-data support.
- `docs/wiki/` — historical notes about installation, data and training.
- `visualizations/` — earlier saved charts; no independently verified reproducibility claim.
- `data_acquisition.md` — instructions on external data sources.

The former README described `models/`, `data/`, dedicated SSL/fine-tuning YAML files and a notebook that are **not present in the current tracked tree**. Do not infer a functioning end-to-end model from that documentation.

## Local source checks

```bash
git clone https://github.com/BalajiKoushik01/multimodel.git
cd multimodel
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m compileall -q scripts utils
```

Read the required imports and arguments in `scripts/train_ssl.py`, `scripts/train_finetune.py` and `scripts/evaluate_model.py` before starting any experiment. Training may not run until missing model components, checkpoint dependencies and dataset files are restored. The current GitHub CI compiles Python source and parses YAML configuration; it does **not** run training or claim accuracy.

## Evaluation and ethics

Before sharing any research result, establish dataset licensing, patient-level train/test separation, exact class definitions, independent validation, confidence intervals and reproducible checkpoints. Existing plots are not a substitute for the underlying dataset, metrics and methods. Protect all patient-identifiable material and never use this code as a medical device.

See [research reproducibility](docs/REPRODUCIBILITY.md).

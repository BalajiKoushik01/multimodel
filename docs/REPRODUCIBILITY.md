# Research reproducibility and completeness

## Source gaps observed
- The current tracked repository includes scripts and utilities but lacks the `models/` and `data/` directories described by the previous README.
- The present configuration file is `configs/config.yaml`. Dedicated `ssl_config.yaml` and `finetune_config.yaml` files referenced in earlier instructions are absent.
- The README previously referred to a visualisation notebook that is not tracked.
- Published figures and logs are historical outputs, not a verified clinical evaluation.

## Before training is reproducible
1. Reconcile imports and dependencies across all training/inference entry points; restore missing modules deliberately, not with empty stubs.
2. Record dataset owners, permissible use, ethical approval (where required), exact checksums and acquisition process.
3. Split by patient and acquisition site where appropriate, preventing correlated images from leaking across train/test.
4. Define labels, mapping of abnormality/severity and sampling strategy.
5. Pin package/CUDA environment, seed and deterministic settings.
6. Archive configuration, checkpoint hash and versioned preprocessing for each experiment.
7. Calculate sensitivity/specificity and uncertainty, use meaningful baselines and external datasets, record failures.
8. Get independent clinical and regulatory validation before any real-world medical use.

**Current state:** exploratory, with source-level CI but no independently reproduced model-quality claim.

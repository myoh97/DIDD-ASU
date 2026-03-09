# [ICLR 2026] Asymmetric Synthetic Data Update for Domain Incremental Dataset Distillation

## Overview
The method builds on M3D (Maximum Mean Discrepancy-based distribution matching) and introduces an asymmetric update mechanism that balances **stability** (preserving knowledge from previous domains) and **plasticity** (adapting to new domains) at the per-image tile level.

## Project Structure

```
DIDD-ASU/
├── dil_condense_m3d.py          # Main condensation script
├── m3dloss.py                   # M3D loss (MMD with kernel methods)
├── util.py                      # Synthesizer, augmentation, evaluation helpers
├── data.py                      # Data loading & transforms
├── model_dist.py                # Model factory (ConvNet, ResNet, etc.)
├── train.py / test.py           # Training & evaluation utilities
├── evaluate_synset.py           # Standalone evaluation for condensed data
├── models/                      # Network architectures
├── dil/datasets/                # Continual learning dataset wrappers
├── configs/                     # YAML configs per dataset/IPC
│   ├── PACS/
│   ├── R-MNIST/
│   └── CORe50-S/
└── scripts/                     # Run scripts
    ├── PACS/run.sh
    ├── R-MNIST/run.sh
    └── CORe50-S/run.sh
```

## Getting Started

### 1. Configuration

Edit the config file for your target dataset in `configs/<dataset>/IPC<n>.yaml`:
- Set `results_path` to your desired output directory
- Adjust hyperparameters as needed (kernel type, learning rates, etc.)

### 2. Run Condensation

Use the provided run scripts:

```bash
# PACS
bash scripts/PACS/run.sh

# Rotated MNIST
bash scripts/R-MNIST/run.sh

# CORe50-S
bash scripts/CORe50-S/run.sh
```

Each script runs condensation across IPC 1, 10, 20 with asymmetric updates enabled. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python dil_condense_m3d.py \
    --ipc=10 \
    --dataset=PACS \
    --asym \
    --lambda_alpha 1e-4 \
    --lambda_beta 1e-4 \
    --flag=asym_1e-4_1e-4
```

### 3. Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name (`PACS`, `R-MNIST`, `CORe50-S`) | `PACS` |
| `--ipc` | Images per class | `1` |
| `--asym` | Enable asymmetric alpha/beta update | `False` |
| `--alpha_lo / --alpha_hi` | Stability coefficient (alpha) bounds | `0.0 / 2.0` |
| `--beta_lo / --beta_hi` | Plasticity coefficient (beta) bounds | `0.0 / 2.0` |
| `--lr_alpha / --lr_beta` | Meta learning rates for alpha/beta | `1e-2` |
| `--lambda_alpha / --lambda_beta` | Sum penalty weights for alpha/beta | `1e-4` |
| `--f_niter` | Override iteration count from config | `3000` |
| `--flag` | Experiment tag (used for save directory) | `None` |


## Acknowledgement

Our code is built upon [M3D](https://arxiv.org/abs/2312.15927). Thanks!

# CleanCam: A Dataset and Benchmark for Camera Lens Cleanliness Classification

## Overview

This repository contains the official analysis pipeline for the CleanCam dataset paper. CleanCam is a real-world dataset for ordinal classification of camera lens cleanliness, comprising field-captured images annotated across five severity levels (Label 1: clean to Label 5: severely dirty). The pipeline supports dataset characterization, integrity auditing, annotation agreement analysis, synthetic data analysis, and CNN benchmarking with standard and ordinal regression methods.

---

## Dataset

### Structure

The dataset is organized into real and synthetic subsets with official train/validation/test splits partitioned at the `capture_id` level to prevent leakage.

```
CleanCam_release/
├── images/
│   ├── real/                        # Field-captured images
│   └── synthetic/                   # Seamless Ultra-Blur synthetic images
├── metadata/
│   ├── metadata.csv                 # Master metadata (real + synthetic)
│   ├── metadata_real.csv
│   ├── metadata_synthetic.csv
│   ├── dirt_assets_manifest.csv
│   ├── split_summary.csv
│   └── build_summary.json
└── splits/
    └── official/
        ├── train_real_only.csv
        ├── train_real_plus_synthetic.csv
        ├── val_real_only.csv
        ├── val_real_plus_synthetic.csv
        ├── test_real_only.csv
        └── test_real_plus_synthetic.csv
```

### Label Taxonomy

| Label | Description |
|-------|-------------|
| 1 | Clean |
| 2 | Slightly dirty |
| 3 | Moderately dirty |
| 4 | Dirty |
| 5 | Severely dirty |

### Synthetic Data

Synthetic images are generated via the Seamless Ultra-Blur pipeline. Each synthetic image inherits the split membership of its real parent image, ensuring no cross-split leakage. Synthetic images are generated exclusively for labels 3, 4, and 5 to augment underrepresented severity classes.

| Split | Synthetic Count |
|-------|----------------|
| Train | 2,800 |
| Val   | 400 |
| Test  | 400 |

### Benchmark Settings

Four official train/evaluation settings are defined:

| Setting | Train Set | Evaluation Set |
|---------|-----------|----------------|
| `train_real_only__eval_real_only` | Real only | Real only |
| `train_real_plus_synthetic__eval_real_only` | Real + Synthetic | Real only |
| `train_real_only__eval_real_plus_synthetic` | Real only | Real + Synthetic |
| `train_real_plus_synthetic__eval_real_plus_synthetic` | Real + Synthetic | Real + Synthetic |

The first two settings constitute the **primary benchmark**. The latter two assess robustness to synthetic data at evaluation time.

---

## Pipeline

### Stages

The pipeline is organized into five stages:

1. **Dataset Characterization** — Label distributions, split compositions, example image grids, and grouping statistics by camera, session, and capture.
2. **Integrity Auditing** — Split disjointness verification, exact duplicate detection (SHA-256), near-duplicate detection (perceptual hashing), and parent leakage checks.
3. **Annotation Agreement** — Pairwise Cohen's kappa, quadratic weighted kappa, raw agreement rates, and image-level disagreement analysis.
4. **Synthetic Data Analysis** — Low-level statistics (sharpness, contrast, entropy), real vs. synthetic comparisons, parent-child delta analysis, and PCA visualization.
5. **CNN Benchmarking** — Training and evaluation of CNN classifiers with support for standard cross-entropy and ordinal regression losses (CORAL, CORN).

### Package Structure

```
cleancam_pipeline/
├── core/               # Configuration, constants, data loading
├── data/               # PyTorch Dataset, transforms, DataLoaders
├── models/             # Model building, training, evaluation, aggregation
│   ├── builder.py
│   ├── training.py
│   ├── evaluation.py
│   ├── aggregation.py
│   └── ordinal.py      # CORAL and CORN ordinal regression
├── analysis/           # Characterization, integrity, annotation, synthetic
├── visualization/      # Plots and confusion matrices
├── orchestrators/      # Stage runners
└── utils/              # I/O, image processing, metrics, seeding
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, recommended for benchmarking)

### Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `opencv-python`, `coral-pytorch`, and optionally `wandb`.

---

## Usage

### Command-Line Interface

```bash
python cleancam_pipeline.py \
    --release-root <path_to_CleanCam_release> \
    --output-root  <path_to_output_directory> \
    [STAGE FLAGS] \
    [OPTIONS]
```

#### Stage Flags

| Flag | Description |
|------|-------------|
| `--run-all` | Run all stages |
| `--run-characterization` | Dataset characterization |
| `--run-integrity` | Integrity auditing |
| `--run-annotation` | Annotation agreement (requires `--annotation-csv`) |
| `--run-synthetic-analysis` | Synthetic data analysis |
| `--run-benchmark` | CNN benchmarking |

#### Benchmark Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | `mobilenet_v2 resnet18 efficientnet_b0` | Model architectures to evaluate |
| `--seeds` | `42 43 44` | Random seeds for multiple runs |
| `--epochs` | `30` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--learning-rate` | `1e-3` | Learning rate |
| `--weight-decay` | `1e-4` | Weight decay |
| `--patience` | `7` | Early stopping patience |
| `--image-size` | `224` | Input image resolution |
| `--num-workers` | `4` | DataLoader worker processes |
| `--ordinal-methods` | *(none)* | Ordinal loss: `coral`, `corn`, or both |
| `--benchmark-settings` | *(all 4)* | Subset of benchmark settings to run |
| `--disable-weighted-sampler` | — | Disable class-balanced sampling |
| `--enable-class-weights` | — | Apply class weights to cross-entropy loss |
| `--cpu-only` | — | Force CPU execution |
| `--single-gpu` | — | Disable multi-GPU (DataParallel) |
| `--no-amp` | — | Disable automatic mixed precision |
| `--no-save-checkpoints` | — | Do not save model checkpoints |
| `--use-wandb` | — | Enable Weights & Biases logging |
| `--wandb-project` | `cleancam-dataset-paper` | W&B project name |
| `--wandb-entity` | *(none)* | W&B entity |
| `--wandb-mode` | `online` | W&B mode: `online`, `offline`, `disabled` |

### Example Commands

**Run full pipeline:**
```bash
python cleancam_pipeline.py \
    --release-root /data/CleanCam_release \
    --output-root  ./output \
    --run-all
```

**Run primary benchmark settings only:**
```bash
python cleancam_pipeline.py \
    --release-root /data/CleanCam_release \
    --output-root  ./output \
    --run-benchmark \
    --models mobilenet_v2 resnet18 efficientnet_b0 \
    --seeds 42 43 44 \
    --benchmark-settings \
        train_real_only__eval_real_only \
        train_real_plus_synthetic__eval_real_only
```

**Run with ordinal regression methods:**
```bash
python cleancam_pipeline.py \
    --release-root /data/CleanCam_release \
    --output-root  ./output \
    --run-benchmark \
    --models mobilenet_v2 resnet18 efficientnet_b0 \
    --ordinal-methods coral corn \
    --seeds 42 43 44
```

---

## Models

### Architectures

Three ImageNet-pretrained CNN architectures are evaluated:

| Model | Parameters | Notes |
|-------|-----------|-------|
| MobileNetV2 | ~3.4M | Lightweight, mobile-oriented |
| ResNet-18 | ~11.7M | Standard residual network |
| EfficientNet-B0 | ~5.3M | Compound-scaled efficient network |

All models are fine-tuned end-to-end with the final classification head replaced for the 5-class ordinal task.

### Loss Functions

| Method | Flag | Description |
|--------|------|-------------|
| Cross-Entropy | *(default)* | Standard nominal classification loss |
| CORAL | `--ordinal-methods coral` | Consistent Rank Logits; binary cumulative loss with weight sharing |
| CORN | `--ordinal-methods corn` | Conditional Ordinal Regression; conditional probability chain |

**CORAL** replaces the final linear layer with a `CoralLayer` (from `coral-pytorch`) and optimizes binary cross-entropy over cumulative rank thresholds.

**CORN** replaces the final linear layer with a standard `nn.Linear(in_features, num_classes - 1)` and optimizes the conditional ordinal loss from `coral-pytorch`.

Both ordinal methods output `num_classes - 1` logits and convert predictions to class labels via threshold-based decoding.

### Training Protocol

- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
- Early stopping: based on validation macro-F1 (patience configurable)
- Sampling: class-balanced weighted random sampling (default)
- Mixed precision: AMP enabled by default on CUDA
- Multi-GPU: DataParallel enabled by default when multiple GPUs are available
- Reproducibility: deterministic algorithms enforced per seed (`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`)

---

## Evaluation Metrics

The following metrics are reported per model, setting, and seed, then aggregated (mean ± std) across seeds:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Macro-F1 | Unweighted mean F1 across all 5 classes |
| Quadratic Weighted Kappa | Agreement metric penalizing ordinal distance |
| MAE | Mean absolute error between predicted and true labels |
| Within-1 Accuracy | Fraction of predictions within ±1 label of ground truth |
| Per-class Precision / Recall / F1 | Per-label breakdown |
| Binary Precision / Recall / F1 | Clean (L1) vs. dirty (L2–L5) |
| Binary AUROC / AUPRC | Binary discrimination metrics |

---

## Outputs

All outputs are written to the specified `--output-root` directory:

```
output/
├── tables/
│   ├── benchmark_summary_main.csv / .tex
│   ├── benchmark_summary_per_class.csv
│   ├── benchmark_summary_binary.csv / .tex
│   ├── benchmark_improvement_summary.csv / .tex
│   ├── benchmark_setting_manifest.csv
│   ├── release_composition.csv
│   ├── integrity_audit.csv
│   └── ...
├── figures/
│   ├── benchmark_macro_f1.png
│   ├── benchmark_weighted_kappa.png
│   ├── benchmark_mae.png
│   ├── confusion_mean_<model>_<setting>.png
│   ├── label_distribution_overall.png
│   └── ...
├── summaries/
│   ├── benchmark_summary.json
│   ├── characterization_summary.json
│   ├── integrity_summary.json
│   ├── environment_summary.json
│   └── ...
└── benchmark/
    └── <model>[_<ordinal_method>]/
        └── <setting>/
            └── seed_<N>/
                ├── best_<model>_<setting>_seed<N>.pt
                ├── train_log_<model>_<setting>_seed<N>.csv
                ├── test_predictions_<model>_<setting>_seed<N>.csv
                └── confusion_matrix_test_norm.png
```

---

## Reproducibility

All experiments are fully reproducible given the same seed. The following measures are applied:

- `random.seed(seed)`
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

Note: DataParallel across multiple GPUs may introduce minor floating-point non-determinism. Use `--single-gpu` for strict reproducibility.

---

## Experiment Tracking

Weights & Biases integration is available via `--use-wandb`. Each run is logged with a unique name encoding the model, ordinal method, benchmark setting, and seed:

```
<prefix>-<model>[-<ordinal_method>]-<setting>-seed<N>
```

Runs are grouped by model and setting for easy comparison in the W&B dashboard.

---

## References

- **CORAL:** Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation. *Pattern Recognition Letters*, 140, 325–331. [arXiv:1901.07884](https://arxiv.org/abs/1901.07884)
- **CORN:** Shi, X., Cao, W., & Raschka, S. (2021). Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities. [arXiv:2111.08851](https://arxiv.org/abs/2111.08851)
- **coral-pytorch:** [https://raschka-research-group.github.io/coral-pytorch/](https://raschka-research-group.github.io/coral-pytorch/)
- **MobileNetV2:** Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.
- **ResNet:** He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- **EfficientNet:** Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

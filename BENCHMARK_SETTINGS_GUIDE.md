# Benchmark Settings Selection Guide

## Overview

You can now select which benchmark settings to run using the `--benchmark-settings` argument.

## Available Settings

1. **train_real_only__eval_real_only** (Primary)
   - Train on real images only
   - Evaluate on real images only

2. **train_real_only__eval_real_plus_synthetic** (Robustness)
   - Train on real images only
   - Evaluate on real + synthetic images

3. **train_real_plus_synthetic__eval_real_only** (Primary)
   - Train on real + synthetic images
   - Evaluate on real images only

4. **train_real_plus_synthetic__eval_real_plus_synthetic** (Robustness)
   - Train on real + synthetic images
   - Evaluate on real + synthetic images

## Usage

### Run All Settings (Default)

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models mobilenet_v2
```

Runs all 4 settings.

### Run Only Primary Settings

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models mobilenet_v2 \
    --benchmark-settings train_real_only__eval_real_only train_real_plus_synthetic__eval_real_only
```

Runs only the 2 primary settings.

### Run Single Setting

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models mobilenet_v2 \
    --benchmark-settings train_real_only__eval_real_only
```

Runs only 1 setting.

## Your Requested Configuration

To run only the two primary settings:

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models mobilenet_v2 resnet18 efficientnet_b0 \
    --ordinal-methods coral corn \
    --seeds 42 43 44 \
    --benchmark-settings train_real_only__eval_real_only train_real_plus_synthetic__eval_real_only \
    --epochs 30 \
    --batch-size 256 \
    --num-workers 2 \
    --use-wandb
```

**Total runs:** 2 settings × 3 models × 2 methods × 3 seeds = **36 runs**

## Run Count Calculation

```
Total Runs = num_settings × num_models × num_methods × num_seeds
```

### Examples

**All settings:**
- 4 settings × 1 model × 1 method × 3 seeds = 12 runs

**Primary settings only:**
- 2 settings × 1 model × 1 method × 3 seeds = 6 runs

**Single setting:**
- 1 setting × 1 model × 1 method × 3 seeds = 3 runs

**Your config (2 primary settings):**
- 2 settings × 3 models × 2 methods × 3 seeds = 36 runs

## Python API

```python
from cleancam_pipeline import BenchmarkConfig

# Run only primary settings
config = BenchmarkConfig(
    models=("mobilenet_v2", "resnet18"),
    ordinal_methods=("coral", "corn"),
    seeds=(42, 43, 44),
    benchmark_settings=(
        "train_real_only__eval_real_only",
        "train_real_plus_synthetic__eval_real_only"
    )
)
```

## Output Structure

With 2 settings, you'll get:

```
output/benchmark/
├── mobilenet_v2_coral/
│   ├── train_real_only__eval_real_only/
│   │   ├── seed_42/
│   │   ├── seed_43/
│   │   └── seed_44/
│   └── train_real_plus_synthetic__eval_real_only/
│       ├── seed_42/
│       ├── seed_43/
│       └── seed_44/
├── mobilenet_v2_corn/
│   └── ... (same structure)
└── ... (other models)
```

## Time Estimation

Assuming 30 minutes per run with GPU:

- **All 4 settings:** 4 × 3 × 2 × 3 = 72 runs = 36 hours
- **Primary 2 settings:** 2 × 3 × 2 × 3 = 36 runs = 18 hours
- **Single setting:** 1 × 3 × 2 × 3 = 18 runs = 9 hours

## Common Configurations

### Quick Test (1 setting, 1 model, 1 seed)

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models mobilenet_v2 \
    --seeds 42 \
    --benchmark-settings train_real_only__eval_real_only \
    --epochs 5
```

Total: 1 run (~30 min)

### Primary Evaluation (2 settings, 3 models, 3 seeds)

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models mobilenet_v2 resnet18 efficientnet_b0 \
    --seeds 42 43 44 \
    --benchmark-settings train_real_only__eval_real_only train_real_plus_synthetic__eval_real_only
```

Total: 18 runs (~9 hours)

### Full Comparison (2 settings, 3 models, 2 methods, 3 seeds)

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models mobilenet_v2 resnet18 efficientnet_b0 \
    --ordinal-methods coral corn \
    --seeds 42 43 44 \
    --benchmark-settings train_real_only__eval_real_only train_real_plus_synthetic__eval_real_only
```

Total: 36 runs (~18 hours)

# Multiple Runs Guide: Seeds, Models, and Ordinal Methods

## Overview

The CleanCam pipeline supports running multiple experiments in a single command by specifying multiple values for:
- **Seeds** (`--seeds`)
- **Models** (`--models`)
- **Ordinal Methods** (`--ordinal-methods`)

All combinations are executed automatically with proper nested looping.

## How It Works

### Nested Loop Structure

```
for setting in [4 benchmark settings]:
    for model in [specified models]:
        for ordinal_method in [specified methods]:
            for seed in [specified seeds]:
                train_one_model()
```

### Total Runs Calculation

```
Total Runs = num_settings × num_models × num_methods × num_seeds
```

## Examples

### Example 1: Single Model, Single Method, Multiple Seeds

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models mobilenet_v2 \
    --ordinal-methods coral \
    --seeds 42 43 44
```

**Runs:**
- 4 settings × 1 model × 1 method × 3 seeds = **12 runs**

**Output:**
```
output/benchmark/mobilenet_v2_coral/
├── train_real_only__eval_real_only/seed_42/
├── train_real_only__eval_real_only/seed_43/
├── train_real_only__eval_real_only/seed_44/
├── train_real_plus_synthetic__eval_real_only/seed_42/
├── train_real_plus_synthetic__eval_real_only/seed_43/
├── train_real_plus_synthetic__eval_real_only/seed_44/
└── ... (2 more settings)
```

### Example 2: Multiple Models, Multiple Methods, Multiple Seeds

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models mobilenet_v2 resnet18 efficientnet_b0 \
    --ordinal-methods coral corn \
    --seeds 42 43 44
```

**Runs:**
- 4 settings × 3 models × 2 methods × 3 seeds = **72 runs**

**Output directories:**
```
output/benchmark/
├── mobilenet_v2_coral/
├── mobilenet_v2_corn/
├── resnet18_coral/
├── resnet18_corn/
├── efficientnet_b0_coral/
└── efficientnet_b0_corn/
```

Each with 4 settings × 3 seeds = 12 subdirectories.

### Example 3: Baseline + Ordinal Methods

To compare baseline (no ordinal method) with CORAL and CORN:

```bash
# Run 1: Baseline
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output_baseline \
    --run-benchmark \
    --models mobilenet_v2 \
    --seeds 42 43 44

# Run 2: Ordinal methods
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output_ordinal \
    --run-benchmark \
    --models mobilenet_v2 \
    --ordinal-methods coral corn \
    --seeds 42 43 44
```

**Total runs:** 12 (baseline) + 24 (ordinal) = 36 runs

## Weights & Biases Integration

When using `--use-wandb`, each run gets a unique name including:
- Model name
- Ordinal method (if specified)
- Setting name
- Seed

### W&B Run Names

```
cleancam-mobilenet_v2-train_real_only__eval_real_only-seed42
cleancam-mobilenet_v2-coral-train_real_only__eval_real_only-seed42
cleancam-mobilenet_v2-corn-train_real_only__eval_real_only-seed42
cleancam-resnet18-coral-train_real_only__eval_real_only-seed42
...
```

### W&B Groups

Runs are grouped by model and method:
```
mobilenet_v2-train_real_only__eval_real_only
mobilenet_v2-coral-train_real_only__eval_real_only
mobilenet_v2-corn-train_real_only__eval_real_only
resnet18-coral-train_real_only__eval_real_only
...
```

This allows easy filtering and comparison in W&B dashboard.

## Argument Syntax

### Seeds

```bash
# Single seed
--seeds 42

# Multiple seeds
--seeds 42 43 44

# Default (if not specified)
--seeds 42 43 44
```

### Models

```bash
# Single model
--models mobilenet_v2

# Multiple models
--models mobilenet_v2 resnet18 efficientnet_b0

# Default (if not specified)
--models mobilenet_v2 resnet18 efficientnet_b0
```

### Ordinal Methods

```bash
# No ordinal method (baseline, cross-entropy)
# (omit --ordinal-methods)

# Single method
--ordinal-methods coral

# Multiple methods
--ordinal-methods coral corn
```

## Output Organization

### Directory Structure

```
output/
├── benchmark/
│   ├── mobilenet_v2_coral/
│   │   ├── train_real_only__eval_real_only/
│   │   │   ├── seed_42/
│   │   │   │   ├── best_mobilenet_v2_*.pt
│   │   │   │   ├── train_log_*.csv
│   │   │   │   ├── test_predictions_*.csv
│   │   │   │   └── confusion_matrix_test_norm.png
│   │   │   ├── seed_43/
│   │   │   └── seed_44/
│   │   ├── train_real_plus_synthetic__eval_real_only/
│   │   └── ...
│   ├── mobilenet_v2_corn/
│   ├── resnet18_coral/
│   └── ...
├── tables/
│   ├── benchmark_summary_main.csv
│   ├── benchmark_summary_per_class.csv
│   └── ...
└── summaries/
    └── benchmark_summary.json
```

### File Naming

Files include model name, setting, and seed:
```
best_mobilenet_v2_train_real_only__eval_real_only_seed42.pt
train_log_mobilenet_v2_train_real_only__eval_real_only_seed42.csv
test_predictions_mobilenet_v2_train_real_only__eval_real_only_seed42.csv
```

## Aggregation and Results

### Automatic Aggregation

After all runs complete, results are automatically aggregated:

```
benchmark_summary_main.csv
├── model (e.g., mobilenet_v2_coral)
├── setting (e.g., train_real_only__eval_real_only)
├── accuracy_mean, accuracy_std
├── macro_f1_mean, macro_f1_std
├── weighted_kappa_mean, weighted_kappa_std
├── mae_mean, mae_std
└── within1_mean, within1_std
```

### Comparison

The summary table allows easy comparison:
```
model                    | setting                      | accuracy_mean | macro_f1_mean | mae_mean
mobilenet_v2             | train_real_only__eval_real   | 0.8234        | 0.7891        | 0.3421
mobilenet_v2_coral       | train_real_only__eval_real   | 0.8312        | 0.8012        | 0.2987
mobilenet_v2_corn        | train_real_only__eval_real   | 0.8401        | 0.8156        | 0.2654
resnet18                 | train_real_only__eval_real   | 0.8156        | 0.7823        | 0.3654
resnet18_coral           | train_real_only__eval_real   | 0.8234        | 0.7945        | 0.3421
resnet18_corn            | train_real_only__eval_real   | 0.8312        | 0.8067        | 0.3012
```

## Troubleshooting

### Only One Run Executes

**Problem:** You specify multiple seeds/methods but only one runs.

**Causes:**
1. Training crashes after first run (check error messages)
2. Early stopping triggered (check patience setting)
3. Argument not parsed correctly

**Solution:**
```bash
# Check arguments are parsed correctly
python cleancam_pipeline.py --help | grep -A 2 "seeds\|models\|ordinal"

# Run with verbose output
python cleancam_pipeline.py ... --log-interval 1
```

### Runs Stop Unexpectedly

**Problem:** Training starts but stops after a few runs.

**Causes:**
1. GPU out of memory
2. Disk space full
3. Keyboard interrupt (Ctrl+C)
4. Timeout on Kaggle

**Solution:**
```bash
# Reduce batch size
--batch-size 128

# Reduce number of workers
--num-workers 2

# Check disk space
df -h

# Run on CPU if GPU issues
--cpu-only
```

### W&B Runs Not Showing

**Problem:** Runs complete but don't appear in W&B dashboard.

**Causes:**
1. `--use-wandb` not specified
2. Not logged in to W&B
3. Wrong project name

**Solution:**
```bash
# Login to W&B
wandb login

# Verify project
python cleancam_pipeline.py ... --use-wandb --wandb-project my-project
```

## Performance Tips

### For Multiple Seeds

- Use same model/method with different seeds for statistical significance
- Typically 3-5 seeds is sufficient
- Seeds should be different but reproducible

### For Multiple Methods

- Compare ordinal methods (CORAL, CORN) with baseline
- Run all methods together for fair comparison
- Same hardware/settings for all methods

### For Multiple Models

- Compare different architectures
- Run all models with same seeds for fair comparison
- Larger models (ResNet) take longer than smaller (MobileNet)

## Example: Complete Comparison

```bash
python cleancam_pipeline.py \
    --release-root /path/to/CleanCam_release \
    --output-root ./output_complete \
    --run-benchmark \
    --models mobilenet_v2 resnet18 efficientnet_b0 \
    --ordinal-methods coral corn \
    --seeds 42 43 44 \
    --epochs 30 \
    --batch-size 256 \
    --num-workers 2 \
    --use-wandb \
    --wandb-project cleancam-comparison
```

**This runs:**
- 4 settings × 3 models × 2 methods × 3 seeds = **72 training runs**
- Estimated time: 72 × 30 min = 36 hours (with GPU)
- All results aggregated in `benchmark_summary_main.csv`
- All runs tracked in W&B with proper naming and grouping

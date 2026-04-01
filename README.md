# CleanCam Dataset Paper Pipeline

A production-ready, modular pipeline for CleanCam dataset characterization, integrity auditing, annotation agreement analysis, synthetic data analysis, and CNN benchmarking.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Python Library](#python-library)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This pipeline provides comprehensive tools for analyzing the CleanCam dataset and benchmarking CNN models for camera lens cleanliness classification (5-class ordinal problem: labels 1-5 from clean to severely dirty).

### What It Does

1. **Dataset Characterization** - Statistics, distributions, visualizations
2. **Integrity Auditing** - Split validation, duplicate detection, leakage checks
3. **Annotation Agreement** - Inter-annotator metrics (Cohen's kappa, etc.)
4. **Synthetic Analysis** - Parent-child comparisons, low-level statistics
5. **CNN Benchmarking** - Train and evaluate ResNet18, MobileNetV2, EfficientNet-B0

---

## ✨ Features

### Dataset Analysis
- Label distribution analysis across subsets
- Split composition tables (train/val/test)
- Grouping by camera, session, day, capture
- Example image grids
- Synthetic source-to-target analysis

### Integrity Audits
- Split disjointness verification
- Exact duplicate detection (SHA256)
- Near-duplicate detection (perceptual hashing)
- Parent leakage checks
- Asset split consistency

### Annotation Agreement
- Pairwise Cohen's kappa
- Quadratic weighted kappa
- Raw agreement rates
- Confusion matrices
- Image-level disagreement analysis

### Synthetic Data Analysis
- Low-level statistics (sharpness, contrast, entropy)
- Real vs synthetic comparisons
- Parent-child delta analysis
- PCA visualization
- Parameter distribution analysis

### CNN Benchmarking
- Multiple architectures (ResNet18, MobileNetV2, EfficientNet-B0)
- Multiple seeds for statistical significance
- Four benchmark settings (primary + robustness)
- Comprehensive metrics (accuracy, F1, kappa, MAE, within-1)
- Per-class and binary metrics
- Confusion matrices
- Weights & Biases integration

---

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU training)

### Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd cleancam-pipeline

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check CLI
python cleancam_pipeline_complete.py --help
```

---

## 🚀 Quick Start

### 1. Run Full Pipeline

```bash
python cleancam_pipeline_complete.py \
    --release-root /path/to/CleanCam_release \
    --output-root ./output \
    --run-all
```

### 2. Run Specific Stages

```bash
# Only characterization
python cleancam_pipeline_complete.py \
    --release-root /path/to/CleanCam_release \
    --output-root ./output \
    --run-characterization

# Only benchmark with custom settings
python cleancam_pipeline_complete.py \
    --release-root /path/to/CleanCam_release \
    --output-root ./output \
    --run-benchmark \
    --models resnet18 mobilenet_v2 \
    --epochs 30 \
    --batch-size 64 \
    --seeds 42 43 44
```

### 3. Use as Python Library

```python
from pathlib import Path
from cleancam_pipeline import BenchmarkConfig, CleanCamRelease
from cleancam_pipeline.orchestrators import run_benchmark
from cleancam_pipeline.utils import OutputManager

# Load data
release = CleanCamRelease(Path("CleanCam_release"))

# Configure
config = BenchmarkConfig(
    epochs=30,
    batch_size=32,
    learning_rate=1e-3,
    models=("resnet18",),
    seeds=(42,)
)

# Run benchmark
output = OutputManager(Path("output"))
run_benchmark(release, output, config)
```

---

## 📖 Usage

### Command Line Interface

#### Basic Options

```bash
python cleancam_pipeline_complete.py \
    --release-root PATH          # Path to CleanCam_release directory (required)
    --output-root PATH           # Path to output directory (required)
    --annotation-csv PATH        # Optional annotation CSV for agreement analysis
```

#### Stage Selection

```bash
--run-all                        # Run all stages
--run-characterization           # Dataset characterization
--run-integrity                  # Integrity audits
--run-annotation                 # Annotation agreement (requires --annotation-csv)
--run-synthetic-analysis         # Synthetic data analysis
--run-benchmark                  # CNN benchmarking
```

#### Integrity Options

```bash
--run-near-duplicate-audit       # Enable near-duplicate detection
--near-duplicate-cap 1000        # Max images to check per split
```

#### Synthetic Analysis Options

```bash
--synthetic-analysis-cap 2000    # Max images to analyze
```

#### Benchmark Options

```bash
# Model selection
--models resnet18 mobilenet_v2 efficientnet_b0

# Training configuration
--epochs 30                      # Number of training epochs
--batch-size 32                  # Batch size
--learning-rate 0.001            # Learning rate
--weight-decay 0.0001            # Weight decay
--patience 7                     # Early stopping patience
--seeds 42 43 44                 # Random seeds for multiple runs

# Data loading
--num-workers 4                  # DataLoader workers
--image-size 224                 # Input image size
--prefetch-factor 4              # Prefetch factor

# Training options
--disable-weighted-sampler       # Disable weighted sampling
--enable-class-weights           # Use class weights in loss
--cpu-only                       # Force CPU training
--no-save-checkpoints            # Don't save model checkpoints
--no-amp                         # Disable mixed precision
--no-persistent-workers          # Disable persistent workers

# Logging
--log-interval 25                # Log every N batches
--use-wandb                      # Enable Weights & Biases
--wandb-project PROJECT          # W&B project name
--wandb-entity ENTITY            # W&B entity
--wandb-mode online              # W&B mode (online/offline/disabled)
--wandb-run-prefix PREFIX        # W&B run name prefix
```

### Python Library

#### Load and Explore Data

```python
from cleancam_pipeline import CleanCamRelease
from pathlib import Path

# Load release
release = CleanCamRelease(Path("CleanCam_release"))

# Access metadata
print(f"Total images: {len(release.metadata)}")
print(f"Real images: {len(release.metadata_real)}")
print(f"Synthetic images: {len(release.metadata_synth)}")

# Access splits
train = release.get_split("train_real_only")
print(f"Training samples: {len(train)}")

# Access benchmark settings
setting = release.get_setting("train_real_only__eval_real_only")
print(f"Train: {len(setting['train'])}")
print(f"Val: {len(setting['val'])}")
print(f"Test: {len(setting['test'])}")
```

#### Custom Training

```python
from cleancam_pipeline import BenchmarkConfig
from cleancam_pipeline.data import build_transforms, make_train_loader
from cleancam_pipeline.models import build_model, evaluate_model
import torch

# Configure
config = BenchmarkConfig(batch_size=64, epochs=10)

# Build data pipeline
train_tf, eval_tf = build_transforms(config.image_size)
train_loader = make_train_loader(train_df, train_tf, config)
test_loader = make_eval_loader(test_df, eval_tf, config)

# Build model
model = build_model("resnet18")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Evaluate
metrics = evaluate_model(model, test_loader, device)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro-F1: {metrics['macro_f1']:.4f}")
```

#### Run Individual Analyses

```python
from cleancam_pipeline.orchestrators import (
    run_characterization,
    run_integrity,
    run_synthetic_analysis
)
from cleancam_pipeline.utils import OutputManager

output = OutputManager(Path("output"))

# Run characterization
run_characterization(release, output)

# Run integrity audit
run_integrity(release, output, 
              run_near_duplicate_audit=True, 
              near_duplicate_cap=1000)

# Run synthetic analysis
run_synthetic_analysis(release, output, cap=2000)
```

---

## 🏗️ Architecture

### Project Structure

```
cleancam-pipeline/
├── cleancam_pipeline/              # Main package
│   ├── core/                       # Core abstractions
│   │   ├── constants.py           # Labels, mappings, settings
│   │   ├── config.py              # Configuration with validation
│   │   └── release.py             # Data loading
│   ├── data/                       # Data pipeline
│   │   ├── dataset.py             # PyTorch Dataset
│   │   ├── transforms.py          # Image preprocessing
│   │   └── loaders.py             # DataLoader creation
│   ├── models/                     # Model components
│   │   ├── builder.py             # Model architectures
│   │   ├── evaluation.py          # Metrics computation
│   │   ├── training.py            # Training loop
│   │   └── aggregation.py         # Result aggregation
│   ├── analysis/                   # Analysis functions
│   │   ├── characterization.py    # Dataset statistics
│   │   ├── integrity.py           # Audit functions
│   │   ├── annotation.py          # Agreement metrics
│   │   └── synthetic.py           # Synthetic analysis
│   ├── visualization/              # Plotting
│   │   ├── plots.py               # All plotting functions
│   │   └── confusion_matrix.py    # CM visualization
│   ├── orchestrators/              # Pipeline runners
│   │   ├── characterization_runner.py
│   │   ├── integrity_runner.py
│   │   ├── annotation_runner.py
│   │   ├── synthetic_runner.py
│   │   └── benchmark_runner.py
│   └── utils/                      # Utilities
│       ├── io.py                  # File I/O
│       ├── image.py               # Image processing
│       ├── metrics.py             # Custom metrics
│       └── seed.py                # Reproducibility
├── tests/                          # Test suite
├── cleancam_pipeline_complete.py  # Main script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

### Module Responsibilities

- **core**: Constants, configuration, data loading
- **data**: PyTorch datasets, transforms, loaders
- **models**: Model building, training, evaluation, aggregation
- **analysis**: Dataset characterization, integrity, annotation, synthetic
- **visualization**: All plotting and visualization functions
- **orchestrators**: High-level pipeline runners
- **utils**: Shared utilities (I/O, image processing, metrics)

---

## ⚙️ Configuration

### BenchmarkConfig

All training parameters are configurable via `BenchmarkConfig`:

```python
from cleancam_pipeline import BenchmarkConfig

config = BenchmarkConfig(
    # Image and batch settings
    image_size=224,
    batch_size=32,
    num_workers=4,
    
    # Training settings
    epochs=30,
    patience=7,
    learning_rate=1e-3,
    weight_decay=1e-4,
    
    # Experiment settings
    seeds=(42, 43, 44),
    models=("mobilenet_v2", "resnet18", "efficientnet_b0"),
    
    # Sampling and weighting
    use_weighted_sampler=True,
    use_class_weights=False,
    
    # Hardware and optimization
    train_on_gpu_if_available=True,
    use_amp=True,
    persistent_workers=True,
    prefetch_factor=4,
    
    # Checkpointing and logging
    save_best_checkpoints=True,
    log_interval=25,
    
    # Weights & Biases
    use_wandb=False,
    wandb_project="cleancam-dataset-paper",
    wandb_entity=None,
    wandb_mode="online",
    wandb_run_prefix="cleancam",
)
```

### Validation

Configuration is validated on creation:

```python
# This will raise ValueError
config = BenchmarkConfig(batch_size=-1)  # batch_size must be positive
config = BenchmarkConfig(epochs=0)       # epochs must be positive
config = BenchmarkConfig(models=())      # models cannot be empty
```

### Benchmark Settings

Four benchmark settings are supported:

1. **train_real_only → eval_real_only** (Primary)
   - Train on real images only
   - Evaluate on real images only

2. **train_real_plus_synthetic → eval_real_only** (Primary)
   - Train on real + synthetic images
   - Evaluate on real images only

3. **train_real_only → eval_real_plus_synthetic** (Robustness)
   - Train on real images only
   - Evaluate on real + synthetic images

4. **train_real_plus_synthetic → eval_real_plus_synthetic** (Robustness)
   - Train on real + synthetic images
   - Evaluate on real + synthetic images

---

## 📤 Outputs

The pipeline generates organized outputs:

```
output/
├── tables/                         # CSV and LaTeX tables
│   ├── release_composition.csv
│   ├── official_split_composition.csv
│   ├── integrity_audit.csv
│   ├── annotation_pairs.csv
│   ├── synthetic_parameter_summary.csv
│   ├── benchmark_summary_main.csv
│   └── ...
├── figures/                        # Plots and visualizations
│   ├── label_distribution_overall.png
│   ├── example_grid_real.png
│   ├── synthetic_vs_real_severe_histograms.png
│   ├── benchmark_macro_f1.png
│   ├── confusion_mean_resnet18_*.png
│   └── ...
├── summaries/                      # JSON summaries
│   ├── characterization_summary.json
│   ├── integrity_summary.json
│   ├── annotation_summary.json
│   ├── synthetic_summary.json
│   ├── benchmark_summary.json
│   └── environment_summary.json
├── logs/                           # Training logs
└── benchmark/                      # Model outputs
    ├── resnet18/
    │   ├── train_real_only__eval_real_only/
    │   │   └── seed_42/
    │   │       ├── best_*.pt                    # Model checkpoint
    │   │       ├── train_log_*.csv              # Training log
    │   │       ├── test_predictions_*.csv       # Predictions
    │   │       └── confusion_matrix_test_norm.png
    │   └── ...
    └── ...
```

### Output Files

#### Tables (CSV + LaTeX)
- Release and split composition
- Integrity audit results
- Annotation agreement metrics
- Synthetic parameter summaries
- Benchmark results (main, per-class, binary)

#### Figures (PNG, 300 DPI)
- Label distributions
- Example image grids
- Real vs synthetic comparisons
- PCA visualizations
- Benchmark comparisons
- Confusion matrices

#### Summaries (JSON)
- Structured data for programmatic access
- Environment information
- Complete results

#### Benchmark Outputs
- Model checkpoints (best epoch)
- Training logs (per epoch)
- Test predictions with probabilities
- Confusion matrices

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=cleancam_pipeline tests/

# Run specific test file
pytest tests/test_core.py -v

# Run specific test
pytest tests/test_core.py::TestBenchmarkConfig::test_invalid_batch_size -v
```

---

## 📚 API Reference

### Core

#### CleanCamRelease
```python
release = CleanCamRelease(release_root: Path)
release.get_split(split_name: str) -> pd.DataFrame
release.get_setting(setting_name: str) -> Dict
```

#### BenchmarkConfig
```python
config = BenchmarkConfig(
    epochs=30,
    batch_size=32,
    # ... see Configuration section
)
config = BenchmarkConfig.from_args(args: argparse.Namespace)
```

### Data

#### Transforms
```python
train_tf, eval_tf = build_transforms(image_size: int)
```

#### Loaders
```python
train_loader = make_train_loader(df, transform, config)
eval_loader = make_eval_loader(df, transform, config)
```

### Models

#### Building
```python
model = build_model(model_name: str, num_classes: int = 5)
# Supported: "resnet18", "mobilenet_v2", "efficientnet_b0"
```

#### Training
```python
result = train_one_setting(
    train_df, val_df, test_df,
    model_name, setting_name, seed,
    cfg, output_dir
)
```

#### Evaluation
```python
metrics = evaluate_model(model, loader, device)
# Returns: accuracy, macro_f1, weighted_kappa, mae, within1,
#          per_class_rows, confusion_matrix, binary_metrics, etc.
```

#### Aggregation
```python
main_df, per_class_df, binary_df, pairwise_df = \
    aggregate_benchmark_results(results)
```

### Orchestrators

```python
from cleancam_pipeline.orchestrators import (
    run_characterization,
    run_integrity,
    run_annotation,
    run_synthetic_analysis,
    run_benchmark
)

run_characterization(release, output_manager)
run_integrity(release, output_manager, run_near_dup, cap)
run_annotation(annotation_csv_path, output_manager)
run_synthetic_analysis(release, output_manager, cap)
run_benchmark(release, output_manager, config)
```

### Utils

```python
from cleancam_pipeline.utils import (
    OutputManager,
    save_csv, save_json, save_table,
    compute_phash, compute_sha256,
    count_by_label, compute_within_one_accuracy,
    set_seed
)
```

---

## 💡 Examples

### Example 1: Quick Benchmark

```python
from pathlib import Path
from cleancam_pipeline import BenchmarkConfig, CleanCamRelease
from cleancam_pipeline.orchestrators import run_benchmark
from cleancam_pipeline.utils import OutputManager

# Setup
release = CleanCamRelease(Path("CleanCam_release"))
output = OutputManager(Path("output"))

# Quick config
config = BenchmarkConfig(
    models=("resnet18",),
    seeds=(42,),
    epochs=10,
    batch_size=64
)

# Run
run_benchmark(release, output, config)
```

### Example 2: Custom Analysis

```python
from cleancam_pipeline.analysis import extract_low_level_stats
from pathlib import Path
import pandas as pd

# Analyze specific images
results = []
for image_path in image_paths:
    stats = extract_low_level_stats(Path(image_path))
    results.append(stats)

df = pd.DataFrame(results)
print(df.describe())
```

### Example 3: Evaluate Existing Model

```python
import torch
from cleancam_pipeline.models import build_model, evaluate_model
from cleancam_pipeline.data import build_transforms, make_eval_loader
from cleancam_pipeline import BenchmarkConfig

# Load model
model = build_model("resnet18")
checkpoint = torch.load("best_model.pt")
model.load_state_dict(checkpoint)

# Setup evaluation
config = BenchmarkConfig()
_, eval_tf = build_transforms(config.image_size)
test_loader = make_eval_loader(test_df, eval_tf, config)

# Evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
metrics = evaluate_model(model, test_loader, device)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro-F1: {metrics['macro_f1']:.4f}")
print(f"Weighted Kappa: {metrics['weighted_kappa']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"Within-1: {metrics['within1']:.4f}")
```

### Example 4: Custom Pipeline

```python
from cleancam_pipeline import CleanCamRelease
from cleancam_pipeline.analysis import make_release_composition_table
from cleancam_pipeline.visualization import plot_label_distribution_overall
from cleancam_pipeline.utils import OutputManager, save_table
from pathlib import Path

# Load data
release = CleanCamRelease(Path("CleanCam_release"))
output = OutputManager(Path("output"))

# Custom analysis
composition = make_release_composition_table(
    release.metadata,
    release.metadata_real,
    release.metadata_synth
)

# Save results
save_table(composition, output.tables_root / "my_composition.csv")

# Create visualization
plot_label_distribution_overall(
    release.metadata,
    release.metadata_real,
    release.metadata_synth,
    output.figures_root / "my_distribution.png"
)
```

---

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 16

# Or use CPU
--cpu-only
```

#### Slow Data Loading
```bash
# Increase workers
--num-workers 8

# Adjust prefetch
--prefetch-factor 8
```

#### Import Errors
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Missing Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
```

#### Weights & Biases Errors
```bash
# Install wandb
pip install wandb

# Login
wandb login

# Or disable
--wandb-mode disabled
```

### Performance Tips

1. **Use GPU** - 10-20x faster training
2. **Enable AMP** - 2x faster with minimal accuracy loss
3. **Increase workers** - Better CPU utilization
4. **Use persistent workers** - Avoid respawning overhead
5. **Batch size** - Larger batches = faster training (if memory allows)

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data loading
from cleancam_pipeline import CleanCamRelease
release = CleanCamRelease(Path("CleanCam_release"))
print(f"Loaded {len(release.metadata)} images")

# Test model building
from cleancam_pipeline.models import build_model
model = build_model("resnet18")
print(model)

# Verify configuration
from cleancam_pipeline import BenchmarkConfig
config = BenchmarkConfig()
print(config)
```

---

## 🤝 Contributing

### Adding New Models

```python
# In cleancam_pipeline/models/builder.py
def build_model(model_name: str, num_classes: int = 5) -> nn.Module:
    if model_name == "your_new_model":
        model = YourModel(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    # ... existing models
```

### Adding New Metrics

```python
# In cleancam_pipeline/utils/metrics.py
def your_custom_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute your custom metric."""
    return float(...)
```

### Adding New Analyses

```python
# Create cleancam_pipeline/analysis/your_analysis.py
def your_analysis_function(data: pd.DataFrame) -> pd.DataFrame:
    """Your custom analysis."""
    return results
```

### Running Tests

```bash
# Before submitting
pytest tests/ -v
```

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

- Original monolithic script authors
- CleanCam dataset creators
- PyTorch and torchvision teams
- scikit-learn contributors

---

## 📞 Support

- **Documentation**: See module docstrings
- **Examples**: Check examples section above
- **Issues**: [GitHub Issues](your-repo-url/issues)
- **Tests**: See `tests/` directory for usage examples

---

## 📊 Statistics

```
Code Organization:
  Modules:        7
  Files:          27
  Lines:          ~3,500 (organized)
  Avg per file:   ~100 lines

Testing:
  Test files:     4
  Unit tests:     41
  Coverage:       ~70%
```

---

**Version**: 2.0.0  
**Status**: Production Ready  
**Python**: 3.8+  
**PyTorch**: 1.10+

---

*Built with ❤️ for reproducible research*

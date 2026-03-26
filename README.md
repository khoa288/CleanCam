# CleanCam public reproducibility repo

This repo is the pipeline needed to reproduce the **public** parts of CleanCam:

1. **Synthetic subset regeneration from the public real release**
2. **Benchmark training and evaluation on the official public split CSVs**

Everything tied to private raw-data curation, filename validation, release rebuilding, copying, and indexing has been removed. That logic was useful for preparing the frozen release, but it is not necessary for a public reproducibility repo.

## Reference links

- **Dataset DOI:** `10.5281/zenodo.18952474`
- **Benchmark report (W&B):** `https://api.wandb.ai/links/khoa288-vinuniversity/2lf2n6h8`

## Scope

The public CleanCam release contains **18,972 real images** and **3,600 synthetic images**, uses deterministic **capture-disjoint official splits**, and defines four official train/eval settings by crossing two training sets with two evaluation domains. The primary benchmark is evaluated on the **real-only** test split; the mixed-domain splits are supplementary robustness settings. The synthetic subset is a constrained training aid and should not be treated as a replacement for real evaluation.

This repo follows those public design choices:

- synthetic generation is done **within each official split only**
- synthetic children never cross parent provenance
- target labels are restricted to **3, 4, 5**
- `3 -> 3` and `1 -> 5` generation are forbidden
- benchmark runs use the official split CSVs under `splits/official/`
- seeds `42 43 44` are supported directly

## Repo layout

```text
.
├── README.md
├── requirements.txt
├── synthetic/
│   └── generate_synthetic.py
└── benchmark/
    └── run_benchmark.py
```

## Expected dataset layout

The scripts assume the downloaded Zenodo release keeps the public structure from the dataset README:

```text
CleanCam/
├── images/
│   ├── real/
│   └── synthetic/
├── metadata/
│   ├── metadata.csv
│   ├── metadata_real.csv
│   ├── metadata_synthetic.csv
│   ├── dirt_assets_manifest.csv
│   └── split_summary.csv
├── splits/
│   └── official/
│       ├── train_real_only.csv
│       ├── train_real_plus_synthetic.csv
│       ├── val_real_only.csv
│       ├── val_real_plus_synthetic.csv
│       ├── test_real_only.csv
│       └── test_real_plus_synthetic.csv
└── assets/
    └── dirt_assets/
```

## Installation

Create a fresh environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## 1) Regenerate the synthetic subset

This script is the cleaned extraction of the synthetic-generation logic from the original release builder. It operates **only on the public dataset layout**. It does **not** re-index raw private images, rename files, or rebuild the entire release.

### What it reads

- `splits/official/train_real_only.csv`
- `splits/official/val_real_only.csv`
- `splits/official/test_real_only.csv`
- `metadata/dirt_assets_manifest.csv`
- real images and dirt assets already present inside the public release

### What it writes

```text
<output-root>/
├── images/synthetic/label_*/...
├── metadata/metadata_synthetic.csv
├── metadata/split_summary.csv
└── splits/official/
    ├── train_synthetic_only.csv
    ├── val_synthetic_only.csv
    ├── test_synthetic_only.csv
    ├── train_real_only.csv
    ├── val_real_only.csv
    ├── test_real_only.csv
    ├── train_real_plus_synthetic.csv
    ├── val_real_plus_synthetic.csv
    └── test_real_plus_synthetic.csv
```

### Default synthetic protocol

- total synthetic images: **3600**
- per-split totals: **train 2800, val 400, test 400**
- target-label totals: **L3 540, L4 900, L5 2160**
- target ratios: `3:0.15, 4:0.25, 5:0.60`
- source policy:
  - target `3` from labels `1` or `2`
  - target `4` from labels `2` or `3`
  - target `5` primarily from label `3`, with label `2` fallback allowed only for training
- max synthetic children per parent:
  - train: `2`
  - val/test: `1`

### Example

```bash
python synthetic/generate_synthetic.py \
  --dataset-root /path/to/CleanCam \
  --output-root /path/to/reproduced_synthetic \
  --seed 42
```

## 2) Run the benchmark

This script trains and evaluates the public baseline benchmark on the official split CSVs.

### Public benchmark protocol encoded here

- models:
  - `mobilenet_v2`
  - `resnet18`
  - `efficientnet_b0`
- image size: **224 × 224**
- batch size: **256**
- learning rate: **1e-3**
- weight decay: **1e-4**
- epochs: up to **30**
- early stopping patience: **7**
- class imbalance handled with a **weighted random sampler**
- reported multiclass metrics:
  - accuracy
  - macro-F1
  - quadratic weighted kappa
  - MAE in label space
  - within-1 accuracy
- reported binary metrics (`Levels 4-5 = Needs cleaning`, `Levels 1-3 = No cleaning yet`):
  - precision
  - recall
  - F1
  - AUROC
  - AUPRC

### Reproduce the primary paper tables

This command runs the primary benchmark setting reported in the paper: both training sets evaluated on the **real-only** validation/test domain, across seeds `42 43 44`.

```bash
python benchmark/run_benchmark.py \
  --dataset-root /path/to/CleanCam \
  --output-root /path/to/benchmark_results \
  --train-splits train_real_only train_real_plus_synthetic \
  --eval-domains real_only \
  --models mobilenet_v2 resnet18 efficientnet_b0 \
  --seeds 42 43 44
```

### Run all four official settings

```bash
python benchmark/run_benchmark.py \
  --dataset-root /path/to/CleanCam \
  --output-root /path/to/benchmark_results_all \
  --train-splits train_real_only train_real_plus_synthetic \
  --eval-domains real_only real_plus_synthetic \
  --models mobilenet_v2 resnet18 efficientnet_b0 \
  --seeds 42 43 44
```

### Outputs

Each run writes its own directory:

```text
<output-root>/
├── train-<train_split>/
│   └── eval-<eval_domain>/
│       └── <model>/
│           └── seed-<seed>/
│               ├── best.pt
│               ├── history.csv
│               ├── metrics.json
│               ├── val_predictions.csv
│               ├── test_predictions.csv
│               └── test_confusion_matrix_normalized.csv
└── summary/
    ├── all_runs.csv
    ├── aggregate_all_settings.csv
    ├── paper_table_5class_real_only.csv
    └── paper_table_binary_real_only.csv
```

## Optional W&B logging

W&B logging is optional.

```bash
python benchmark/run_benchmark.py \
  --dataset-root /path/to/CleanCam \
  --output-root /path/to/benchmark_results \
  --eval-domains real_only \
  --use-wandb \
  --wandb-project cleancam-benchmark \
  --wandb-entity <entity>
```

## License

This repository's **code** is released under the **MIT License**. See `LICENSE`.

The **dataset** is distributed separately through Zenodo under the license declared on the dataset record.

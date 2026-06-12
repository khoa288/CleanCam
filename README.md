# CleanCam: A Labelled Image Dataset for Camera-Cleaning Decisions

CleanCam is a labelled underwater-image dataset for camera-health assessment in aquaculture monitoring. It separates water-column degradation from material attached to the camera viewport, so the label is tied to a camera-cleaning decision rather than generic image quality.

The dataset contains real field images, split-consistent synthetic viewport-fouling images, metadata tables, capture-disjoint train/validation/test splits, and benchmark code for ordinal cleaning-severity classification.

Dataset DOI: https://doi.org/10.5281/zenodo.18952474

---

## Repository contents

```text
CleanCam/
├── cleancam_pipeline/              # Analysis and benchmark package
├── cleancam_pipeline.py            # Main benchmark/analysis CLI
├── scripts/
│   └── build_cleancam_release.py   # Dataset release builder
├── requirements.txt
├── README.md
└── LICENSE
```

This GitHub repository contains code, documentation, and reproducibility scripts. The image dataset is hosted separately on Zenodo.

Do not commit raw local images, generated dataset folders, benchmark outputs, or large archives directly to this repository.

---

## Dataset

The CleanCam dataset is available on Zenodo:

> Nguyen, M. K., Hoang, T. A., Tran, N. N. A., Tran, N. N. A., Pham, M. H., Phan, T. K., Nguyen, V. D., Dinh, V. D., & Do, D. C. (2026). *CleanCam: a labelled image dataset for camera-cleaning decisions in aquaculture monitoring* (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18952474

After downloading and extracting the dataset, the expected structure is:

```text
CleanCam_release/
├── images/
│   ├── real/                        # Curated field-captured images
│   └── synthetic/                   # Split-consistent synthetic viewport-fouling images
├── assets/
│   └── dirt_assets/                 # Deposit assets used by the synthetic generator, if included
├── code/
│   └── build_cleancam_release.py    # Builder script copied into the dataset package
├── metadata/
│   ├── metadata.csv                 # Master metadata table, real + synthetic
│   ├── metadata_real.csv
│   ├── metadata_synthetic.csv
│   ├── dirt_assets_manifest.csv
│   ├── skipped_real_images.csv
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

The official splits are partitioned at the `capture_id` level. Synthetic images inherit the split of their real parent image, which prevents parent-image information from crossing train, validation, and test partitions.

---

## Label taxonomy

| Label | Viewport condition | Cleaning interpretation |
|---:|---|---|
| 1 | Clean viewport. Visibility may still be affected by water, particles, or lighting. | No cleaning is indicated by the image alone. |
| 2 | Light local deposits, with the view still largely usable. | Cleaning is not urgent; continue monitoring. |
| 3 | Clear deposits affecting part of the field of view. | Inspect or clean if the condition persists or conflicts with the monitoring task. |
| 4 | Severe deposits or smearing interfering with routine interpretation. | Cleaning is recommended before routine image-based monitoring continues. |
| 5 | Heavy obstruction or blur that strongly compromises scene interpretation. | Cleaning is required before treating the image stream as reliable. |

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.8+ is recommended. CUDA is optional but useful for CNN benchmarking.

---

## Build the dataset package from local source files

Use `scripts/build_cleancam_release.py` only if you need to rebuild the dataset package from curated local source images and deposit assets.

### Expected local inputs

Keep these folders local. They should not be committed to GitHub.

```text
raw/
├── label_by_cam_artifacts/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── 5/
└── dirt_assets/
```

The real-image filename parser is strict. Files with malformed names, camera/date/timecode inconsistencies, or duplicated logical capture identifiers are excluded and recorded in `metadata/skipped_real_images.csv`.

### Build command

```bash
python scripts/build_cleancam_release.py \
    --dataset-root raw/label_by_cam_artifacts \
    --dirt-assets-dir raw/dirt_assets \
    --release-root CleanCam_release \
    --release-tag v1.0.0 \
    --compute-sha256
```

To inspect all builder options:

```bash
python scripts/build_cleancam_release.py --help
```

Common options include:

| Argument | Purpose |
|---|---|
| `--dataset-root` | Path to curated real images grouped by labels `1` to `5`. |
| `--dirt-assets-dir` | Path to deposit/viewport-fouling assets used for synthetic generation. |
| `--release-root` | Output directory for the generated dataset package. |
| `--release-tag` | Version tag stored in the generated metadata. |
| `--target-synthetic-total` | Total number of synthetic images to generate. |
| `--synthetic-partition-ratios` | Train/validation/test synthetic count ratios. |
| `--synthetic-target-ratios` | Label distribution for synthetic labels 3, 4, and 5. |
| `--synthetic-source-mix` | Parent-label mix for each synthetic target label. |
| `--no-copy-dirt-assets-to-release` | Keep deposit assets out of the generated dataset folder. |
| `--compute-sha256` | Compute image checksums for metadata and auditing. |

---

## Run the analysis and benchmark pipeline

After downloading or building `CleanCam_release/`, run the analysis and benchmark pipeline with:

```bash
python cleancam_pipeline.py \
    --release-root CleanCam_release \
    --output-root output \
    --run-all
```

Run the primary real-domain benchmark settings with:

```bash
python cleancam_pipeline.py \
    --release-root CleanCam_release \
    --output-root output \
    --run-benchmark \
    --models mobilenet_v2 resnet18 efficientnet_b0 \
    --seeds 42 43 44 \
    --benchmark-settings \
        train_real_only__eval_real_only \
        train_real_plus_synthetic__eval_real_only
```

Run ordinal-regression baselines with:

```bash
python cleancam_pipeline.py \
    --release-root CleanCam_release \
    --output-root output \
    --run-benchmark \
    --models mobilenet_v2 resnet18 efficientnet_b0 \
    --ordinal-methods coral corn \
    --seeds 42 43 44
```

---

## Benchmark stages

The pipeline supports five stages:

1. Dataset characterization: label distributions, split composition, example grids, and grouping statistics.
2. Integrity auditing: capture-disjoint split checks, duplicate checks, near-duplicate checks, and synthetic parent-leakage checks.
3. Annotation agreement: pairwise agreement, Cohen's kappa, quadratic weighted kappa, and disagreement summaries.
4. Synthetic data analysis: low-level image statistics, real-versus-synthetic comparisons, parent-child deltas, and PCA visualization.
5. CNN benchmarking: MobileNetV2, ResNet-18, EfficientNet-B0, cross-entropy, CORAL, and CORN.

---

## Main outputs

Pipeline outputs are written under the selected `--output-root`:

```text
output/
├── tables/
├── figures/
├── summaries/
└── benchmark/
```

Common outputs include benchmark summaries, release composition tables, integrity audits, confusion matrices, prediction files, training logs, and model checkpoints.

---

## Reproducibility notes

The benchmark pipeline fixes Python, NumPy, and PyTorch seeds per run. It also enables deterministic PyTorch settings where supported. Multi-GPU execution can still introduce minor floating-point nondeterminism, so use `--single-gpu` when strict reproducibility is required.

For dataset-package reproducibility, preserve:

- the Zenodo dataset version and DOI,
- `metadata/build_summary.json`,
- `code/build_cleancam_release.py` inside the dataset package,
- the GitHub commit or release tag used for the code repository.

---

## Citation

If you use CleanCam, please cite the dataset:

```bibtex
@dataset{nguyen_2026_cleancam,
  author    = {Nguyen, Minh Khoa and Hoang, Tuan Anh and Tran, Nhat-Nam Anh and Tran, Nguyet-Nam Anh and Pham, Minh Hoang and Phan, Tuan Khoi and Nguyen, Van Dinh and Dinh, Van Dung and Do, Dinh Cuong},
  title     = {CleanCam: a labelled image dataset for camera-cleaning decisions in aquaculture monitoring},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.18952474},
  url       = {https://doi.org/10.5281/zenodo.18952474}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

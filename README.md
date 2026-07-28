# CleanCam

CleanCam is a labelled underwater-image dataset for assessing material
attached to camera viewports in aquaculture monitoring. Its five ordered
levels describe cleaning severity rather than generic image quality, helping
separate persistent viewport deposits from turbidity, suspended particles,
lighting changes, and other water-column effects.

The validated v2 release contains:

- 18,972 field-captured RGB JPEG images;
- 3,600 split-consistent synthetic RGB JPEG images for Levels 3–5;
- 22,572 images in total, all at 3072 × 2048 pixels;
- image-level metadata and a complete data dictionary;
- capture-disjoint train, validation, and test splits;
- 11 documented RGBA deposit assets in disjoint split pools; and
- synthetic parentage, transformation parameters, seeds, and checksums.

Dataset v2.0.0 DOI: <https://doi.org/10.5281/zenodo.21515620>

Concept DOI (all versions): <https://doi.org/10.5281/zenodo.18952473>

Current release: v2.0.0

## Repository contents

```text
CleanCam/
├── cleancam_pipeline/                 # Analysis and optional benchmark package
├── configs/
│   └── manuscript_examples.json       # Fixed image IDs used in Figures 2 and 4
├── docs/
│   ├── acquisition_conditions.md
│   ├── data_dictionary.csv
│   ├── deposit_assets.md
│   ├── deposit_assets.csv
│   ├── label_taxonomy.csv
│   ├── manuscript_reproducibility.md
│   └── quickstart.md
├── scripts/
│   ├── build_cleancam_release.py
│   ├── finalize_cleancam_v2.py
│   ├── reproduce_manuscript.py
│   └── validate_release.py
├── requirements.txt                  # Pinned core reproduction environment
├── requirements-benchmark.txt        # Optional PyTorch benchmark stack
└── requirements-dev.txt
```

The image dataset is hosted on Zenodo and is intentionally not stored in this
Git repository.

## Dataset structure

After downloading and extracting `CleanCam_v2.zip`, the top-level structure is:

```text
CleanCam_v2/
├── images/
│   ├── real/
│   └── synthetic/
├── assets/
│   └── dirt_assets/
├── code/
├── documentation/
├── metadata/
│   ├── metadata.csv
│   ├── metadata_real.csv
│   ├── metadata_synthetic.csv
│   ├── dirt_assets_manifest.csv
│   ├── split_summary.csv
│   ├── build_summary.json
│   └── file_manifest_sha256.csv
└── splits/
    └── official/
        ├── train_real_only.csv
        ├── train_real_plus_synthetic.csv
        ├── val_real_only.csv
        ├── val_real_plus_synthetic.csv
        ├── test_real_only.csv
        └── test_real_plus_synthetic.csv
```

Synthetic images inherit the split of their real parent image. Deposit assets
are also partitioned into disjoint train, validation, and test pools.

## Label taxonomy

| Level | Viewport condition | Cleaning interpretation |
|---:|---|---|
| L1 | Clean viewport. Visibility may still be affected by water, particles, or lighting. | No cleaning is indicated by the image alone. |
| L2 | Light local deposits; the view remains largely usable. | Cleaning is not urgent; continue monitoring. |
| L3 | Clear deposits affect part of the field of view. | Inspect or clean if the condition persists or conflicts with the monitoring task. |
| L4 | Severe deposits or smearing interfere with routine interpretation. | Cleaning is recommended before routine image-based monitoring continues. |
| L5 | Heavy obstruction or blur strongly compromises scene interpretation. | Cleaning is required before treating the image stream as reliable. |

The machine-readable definitions are in
[`docs/label_taxonomy.csv`](docs/label_taxonomy.csv).

## Core installation

Python 3.11 is the frozen v2 reproduction environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The core environment reproduces release validation and manuscript outputs
without PyTorch.

## Reproduce the manuscript outputs

```bash
python scripts/reproduce_manuscript.py \
  --release-root data/CleanCam_v2 \
  --output-root output/manuscript
```

The command regenerates Figures 2–6 and Tables 1–3. It also saves the
image-level low-level statistics, PCA loadings, PCA explained variance, fixed
example IDs, package versions, and SHA-256 checksums of all generated outputs.
Figure 1 is a collection-setup source image and is retained at
`docs/manuscript_sources/Figure_1_Data_collection_setup.png`.

See [`docs/manuscript_reproducibility.md`](docs/manuscript_reproducibility.md)
for the item-by-item mapping.

## Validate a release

```bash
python scripts/validate_release.py data/CleanCam_v2
```

For a full image-header pass:

```bash
python scripts/validate_release.py \
  data/CleanCam_v2 \
  --verify-image-files
```

To verify every public file against the release SHA-256 manifest:

```bash
python scripts/validate_release.py \
  data/CleanCam_v2 \
  --verify-sha256
```

## Optional CNN benchmarks

Install the benchmark stack after the core environment:

```bash
python -m pip install -r requirements-benchmark.txt
```

Each supported backbone—MobileNetV2, ResNet-18, and EfficientNet-B0—can use
the default five-class classification head trained using cross-entropy or an
ordinal CORAL or CORN head. Omit `--ordinal-methods` for cross-entropy, or list
one or both ordinal heads.

For example, the following command runs the cross-entropy benchmark:

```bash
python cleancam_pipeline.py \
  --release-root data/CleanCam_v2 \
  --output-root output/benchmark \
  --run-benchmark \
  --models mobilenet_v2 resnet18 efficientnet_b0 \
  --seeds 42 43 44
```

Use `--ordinal-methods coral corn` to run both ordinal variants.

The primary evaluation domain is the real-only validation and test data.

## Build tooling

`scripts/build_cleancam_release.py` reconstructs the full release from curated
source images and RGBA deposit assets. `scripts/finalize_cleancam_v2.py`
assembles the public documentation, asset descriptions, validation manifest,
and deterministic ZIP archive from a validated working tree.

## Citation

```bibtex
@dataset{nguyen_2026_cleancam,
  author    = {Nguyen, Minh Khoa and Hoang, Tuan Anh and Tran, Nam Nhat Anh and Tran, Nam Nguyet Anh and Pham, Minh Hoang and Phan, Tuan Khoi and Nguyen, Van Dinh and Dinh, Van Dung and Do, Danh Cuong},
  title     = {CleanCam: a labelled image dataset for camera-cleaning decisions in aquaculture monitoring},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.0.0},
  doi       = {10.5281/zenodo.21515620},
  url       = {https://doi.org/10.5281/zenodo.21515620}
}
```

## Licenses

The code is released under the MIT License. The dataset release is licensed
under CC BY 4.0.

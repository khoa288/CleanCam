# CleanCam v2 quickstart

CleanCam v2 separates field images from split-consistent synthetic images and
provides capture-disjoint train, validation, and test lists.

## 1. Install the core environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pinned core environment reproduces the release checks, manuscript tables,
and Figures 2–6 without installing PyTorch.

## 2. Download and extract the data

Download `CleanCam_v2.zip` from the CleanCam Zenodo record and extract it. The
examples below assume that the extracted folder is `data/CleanCam_v2`.

## 3. Inspect one official split

```python
from pathlib import Path

import pandas as pd
from PIL import Image

root = Path("data/CleanCam_v2")
split = pd.read_csv(root / "splits/official/train_real_only.csv")
row = split.iloc[0]
image = Image.open(root / row["relative_path"])

print(row[["image_id", "label", "capture_id"]].to_dict())
print(image.size)
```

Labels are ordinal: L1 is a clean viewport and L5 is heavy obstruction. Use
`documentation/label_taxonomy.csv` for the full definitions.

## 4. Reproduce the manuscript outputs

```bash
python scripts/reproduce_manuscript.py \
  --release-root data/CleanCam_v2 \
  --output-root output/manuscript
```

The command writes the plotted figures, the source tables behind them, the
PCA loadings and explained variance, and a SHA-256 output manifest. Figure 1 is
a source photograph rather than a computed figure and is retained at
`docs/manuscript_sources/Figure_1_Data_collection_setup.png`.

## 5. Run release integrity checks

```bash
python scripts/validate_release.py data/CleanCam_v2
```

Add `--verify-image-files` for a full image-header and dimension pass, or
`--verify-sha256` to compare every file with the release manifest.

## Optional benchmark dependencies

The CNN and ordinal-regression workflows are separate from manuscript
reproduction:

```bash
python -m pip install -r requirements-benchmark.txt
```

The optional benchmark entry point fine-tunes MobileNetV2, ResNet-18, or
EfficientNet-B0 on the official capture-disjoint splits. The default is a
five-class classification head trained using cross-entropy; CORAL and CORN
ordinal heads are selected with `--ordinal-methods coral corn`. See the
repository README for the complete command.

"""Constants and mappings for CleanCam dataset."""

from typing import Dict, Tuple

# Label definitions
LABELS = [1, 2, 3, 4, 5]
LABEL_TO_INDEX: Dict[int, int] = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL: Dict[int, int] = {idx: label for label, idx in LABEL_TO_INDEX.items()}

# Benchmark settings
PRIMARY_SETTINGS: Tuple[str, ...] = (
    "train_real_only__eval_real_only",
    "train_real_plus_synthetic__eval_real_only",
)

ALL_SETTINGS: Tuple[str, ...] = (
    "train_real_only__eval_real_only",
    "train_real_only__eval_real_plus_synthetic",
    "train_real_plus_synthetic__eval_real_only",
    "train_real_plus_synthetic__eval_real_plus_synthetic",
)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

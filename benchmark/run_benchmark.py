#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

try:
    from scipy.stats import entropy as scipy_entropy
    from scipy.stats import wasserstein_distance
except Exception:  # pragma: no cover
    scipy_entropy = None
    wasserstein_distance = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

try:
    from coral_pytorch.dataset import (
        corn_label_from_logits,
        levels_from_labelbatch,
        proba_to_label,
    )
    from coral_pytorch.layers import CoralLayer
    from coral_pytorch.losses import coral_loss, corn_loss
except Exception:  # pragma: no cover
    corn_label_from_logits = None
    levels_from_labelbatch = None
    proba_to_label = None
    CoralLayer = None
    coral_loss = None
    corn_loss = None

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


# ============================================================
# Configuration
# ============================================================


OBJECTIVES: Tuple[str, ...] = ("ce", "coral", "corn")


@dataclass
class BenchmarkConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seeds: Tuple[int, ...] = (42, 43, 44)
    models: Tuple[str, ...] = ("mobilenet_v2", "resnet18", "efficientnet_b0")
    objectives: Tuple[str, ...] = ("ce",)
    use_weighted_sampler: bool = True
    use_class_weights: bool = False
    train_on_gpu_if_available: bool = True
    save_best_checkpoints: bool = True
    use_amp: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    log_interval: int = 25
    use_wandb: bool = False
    wandb_project: str = "cleancam-benchmarks"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"
    wandb_run_prefix: str = "cleancam"


# ============================================================
# General utilities
# ============================================================


LABELS = [1, 2, 3, 4, 5]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}
NUM_CLASSES = len(LABELS)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_float(x: float) -> float:
    if pd.isna(x):
        return float("nan")
    return float(x)


def normalize_split_name(name: str) -> str:
    return name.strip().lower().replace(".csv", "")


def load_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def image_to_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(
    path: Path, hash_size: int = 8, highfreq_factor: int = 4
) -> np.ndarray:
    """
    Simple pHash implementation using DCT.
    Returns a flattened boolean hash array.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image for pHash: {path}")
    size = hash_size * highfreq_factor
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = np.float32(img)
    dct = cv2.dct(img)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:]) if dct_low.size > 1 else np.median(dct_low)
    return (dct_low > med).astype(np.uint8).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def require_ordinal_dependencies(objective: str) -> None:
    if objective == "ce":
        return
    missing = []
    if CoralLayer is None:
        missing.append("coral_pytorch.layers.CoralLayer")
    if levels_from_labelbatch is None:
        missing.append("coral_pytorch.dataset.levels_from_labelbatch")
    if proba_to_label is None:
        missing.append("coral_pytorch.dataset.proba_to_label")
    if corn_label_from_logits is None:
        missing.append("coral_pytorch.dataset.corn_label_from_logits")
    if coral_loss is None:
        missing.append("coral_pytorch.losses.coral_loss")
    if corn_loss is None:
        missing.append("coral_pytorch.losses.corn_loss")
    if missing:
        joined = ", ".join(missing)
        raise ImportError(
            "Ordinal objectives require coral-pytorch. "
            "Install it with `pip install coral-pytorch`. "
            f"Missing symbols: {joined}"
        )


def objective_display_name(objective: str) -> str:
    return objective.upper() if objective != "ce" else "CE"


# ============================================================
# Release reader
# ============================================================


class CleanCamRelease:
    def __init__(self, release_root: Path):
        self.release_root = release_root

        self.metadata_root = release_root / "metadata"
        self.splits_root = release_root / "splits"
        self.images_root = release_root / "images"

        self.metadata = read_csv(self.metadata_root / "metadata.csv")
        self.metadata_real = read_csv(self.metadata_root / "metadata_real.csv")
        self.metadata_synth = read_csv(self.metadata_root / "metadata_synthetic.csv")
        self.split_summary = read_csv(self.metadata_root / "split_summary.csv")
        self.build_summary = json.loads(
            (self.metadata_root / "build_summary.json").read_text(encoding="utf-8")
        )

        self.official_splits = {
            "train_real_only": read_csv(
                self.splits_root / "official" / "train_real_only.csv"
            ),
            "train_real_plus_synthetic": read_csv(
                self.splits_root / "official" / "train_real_plus_synthetic.csv"
            ),
            "val": read_csv(self.splits_root / "official" / "val.csv"),
            "test": read_csv(self.splits_root / "official" / "test.csv"),
        }

        self.cv5_splits = self._load_cv5_splits()

        self._prepare_metadata_paths()

    def _load_cv5_splits(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        out: Dict[str, Dict[str, pd.DataFrame]] = {}
        cv_root = self.splits_root / "cv5"
        if not cv_root.exists():
            return out
        for fold_dir in sorted([p for p in cv_root.iterdir() if p.is_dir()]):
            out[fold_dir.name] = {}
            for split_name in [
                "train_real_only",
                "train_real_plus_synthetic",
                "val",
                "test",
            ]:
                split_path = fold_dir / f"{split_name}.csv"
                if split_path.exists():
                    out[fold_dir.name][split_name] = read_csv(split_path)
        return out

    def _prepare_metadata_paths(self) -> None:
        for df in [self.metadata, self.metadata_real, self.metadata_synth]:
            if "relative_path" in df.columns:
                df["absolute_path"] = df["relative_path"].map(
                    lambda rp: str(self.release_root / rp)
                )
            else:
                df["absolute_path"] = df["release_path"]

        for split_df in self.official_splits.values():
            if "relative_path" in split_df.columns:
                split_df["absolute_path"] = split_df["relative_path"].map(
                    lambda rp: str(self.release_root / rp)
                )
            else:
                split_df["absolute_path"] = split_df["release_path"]

        for fold_dict in self.cv5_splits.values():
            for split_df in fold_dict.values():
                if "relative_path" in split_df.columns:
                    split_df["absolute_path"] = split_df["relative_path"].map(
                        lambda rp: str(self.release_root / rp)
                    )
                else:
                    split_df["absolute_path"] = split_df["release_path"]


# ============================================================
# Output manager
# ============================================================


class OutputManager:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.tables_root = output_root / "tables"
        self.figures_root = output_root / "figures"
        self.logs_root = output_root / "logs"
        self.summaries_root = output_root / "summaries"
        self.benchmark_root = output_root / "benchmark"
        ensure_dir(self.tables_root)
        ensure_dir(self.figures_root)
        ensure_dir(self.logs_root)
        ensure_dir(self.summaries_root)
        ensure_dir(self.benchmark_root)


# ============================================================
# Characterization
# ============================================================


def make_composition_table(release: CleanCamRelease) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for name, df in [
        ("real", release.metadata_real),
        ("synthetic", release.metadata_synth),
        ("all", release.metadata),
    ]:
        row = {"subset": name, "n_images": int(len(df))}
        counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
        for label in LABELS:
            row[f"label_{label}"] = int(counts.loc[label])
        rows.append(row)

    return pd.DataFrame(rows)


def make_official_split_composition_table(release: CleanCamRelease) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split_name, df in release.official_splits.items():
        row = {
            "split": split_name,
            "n_images": int(len(df)),
            "n_real": int((df["origin"] == "real").sum()),
            "n_synthetic": int((df["origin"] == "synthetic").sum()),
        }
        counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
        for label in LABELS:
            row[f"label_{label}"] = int(counts.loc[label])
        rows.append(row)
    return pd.DataFrame(rows)


def make_by_field_table(df: pd.DataFrame, field: str, name: str) -> pd.DataFrame:
    out = (
        df.groupby([field, "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    out.columns = [field] + [f"label_{c}" for c in LABELS]
    out.insert(0, "table_name", name)
    out["n_images"] = out[[f"label_{c}" for c in LABELS]].sum(axis=1)
    return out


def plot_label_distribution_overall(release: CleanCamRelease, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    subsets = [
        ("Real", release.metadata_real),
        ("Synthetic", release.metadata_synth),
        ("All", release.metadata),
    ]
    x = np.arange(len(LABELS))
    width = 0.25

    for i, (name, df) in enumerate(subsets):
        counts = df["label"].value_counts().reindex(LABELS, fill_value=0).to_numpy()
        plt.bar(x + (i - 1) * width, counts, width=width, label=name)

    plt.xticks(x, [f"L{label}" for label in LABELS])
    plt.ylabel("Number of images")
    plt.title("CleanCam label distribution")
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_label_distribution_official_splits(
    release: CleanCamRelease, out_path: Path
) -> None:
    splits = ["train_real_only", "train_real_plus_synthetic", "val", "test"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, split_name in zip(axes, splits):
        df = release.official_splits[split_name]
        counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
        ax.bar([f"L{label}" for label in LABELS], counts.values)
        ax.set_title(split_name)
        ax.set_ylabel("Count")

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_counts_by_day_camera_session(release: CleanCamRelease, out_path: Path) -> None:
    df = release.metadata_real.copy()
    df["day"] = df["day"].astype(str)
    pivot = df.groupby(["day", "cam_state"]).size().unstack(fill_value=0).sort_index()

    plt.figure(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, figsize=(14, 6))
    plt.ylabel("Number of real images")
    plt.title("Real-image counts by day and camera/session")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_example_grid(
    release: CleanCamRelease,
    out_path: Path,
    seed: int = 42,
    max_per_label: int = 3,
    origin: str = "real",
) -> None:
    rng = random.Random(seed)
    df = release.metadata[release.metadata["origin"] == origin].copy()
    samples: List[pd.DataFrame] = []

    for label in LABELS:
        sub = df[df["label"] == label]
        if len(sub) == 0:
            continue
        chosen_idx = rng.sample(list(sub.index), min(max_per_label, len(sub)))
        samples.append(sub.loc[chosen_idx])

    if not samples:
        return

    chosen = pd.concat(samples, ignore_index=True)
    rows = len(LABELS)
    cols = max_per_label
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    if rows == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(rows, 1)

    for ax in axes.flatten():
        ax.axis("off")

    for label in LABELS:
        sub = chosen[chosen["label"] == label].reset_index(drop=True)
        for j in range(min(cols, len(sub))):
            row = label - 1
            ax = axes[row, j]
            img = load_image_rgb(Path(sub.loc[j, "absolute_path"]))
            ax.imshow(img)
            ax.set_title(f"L{label} | {Path(sub.loc[j, 'relative_path']).name[:40]}")
            ax.axis("off")

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_characterization(release: CleanCamRelease, out: OutputManager) -> None:
    composition = make_composition_table(release)
    official_split_comp = make_official_split_composition_table(release)

    by_cam = make_by_field_table(release.metadata_real, "cam", "counts_by_cam")
    by_state = make_by_field_table(release.metadata_real, "state", "counts_by_state")
    by_day = make_by_field_table(release.metadata_real, "day", "counts_by_day")
    synth_parent = (
        release.metadata_synth.groupby("source_label")
        .size()
        .reset_index(name="n_images")
        if len(release.metadata_synth)
        else pd.DataFrame(columns=["source_label", "n_images"])
    )

    save_csv(composition, out.tables_root / "dataset_composition.csv")
    save_csv(official_split_comp, out.tables_root / "official_split_composition.csv")
    save_csv(by_cam, out.tables_root / "counts_by_camera.csv")
    save_csv(by_state, out.tables_root / "counts_by_session.csv")
    save_csv(by_day, out.tables_root / "counts_by_day.csv")
    save_csv(synth_parent, out.tables_root / "synthetic_parent_label_distribution.csv")

    plot_label_distribution_overall(
        release, out.figures_root / "label_distribution_overall.png"
    )
    plot_label_distribution_official_splits(
        release, out.figures_root / "label_distribution_official_splits.png"
    )
    plot_counts_by_day_camera_session(
        release, out.figures_root / "counts_by_day_camera_session.png"
    )
    make_example_grid(
        release, out.figures_root / "example_grid_by_label_real.png", origin="real"
    )
    if len(release.metadata_synth) > 0:
        make_example_grid(
            release,
            out.figures_root / "example_grid_by_label_synthetic.png",
            origin="synthetic",
        )

    save_json(
        {
            "dataset_composition": composition.to_dict(orient="records"),
            "official_split_composition": official_split_comp.to_dict(orient="records"),
        },
        out.summaries_root / "characterization_summary.json",
    )


# ============================================================
# Split integrity / leakage audit
# ============================================================


def audit_official_split_integrity(release: CleanCamRelease) -> pd.DataFrame:
    train_real = release.official_splits["train_real_only"]
    train_aug = release.official_splits["train_real_plus_synthetic"]
    val_df = release.official_splits["val"]
    test_df = release.official_splits["test"]

    rows: List[Dict[str, object]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        rows.append({"check": name, "passed": bool(passed), "details": details})

    def overlap(a: Iterable[str], b: Iterable[str]) -> set:
        return set(a).intersection(set(b))

    add_check(
        "no_image_id_overlap_train_val",
        len(overlap(train_real["image_id"], val_df["image_id"])) == 0,
        f"overlap={len(overlap(train_real['image_id'], val_df['image_id']))}",
    )
    add_check(
        "no_image_id_overlap_train_test",
        len(overlap(train_real["image_id"], test_df["image_id"])) == 0,
        f"overlap={len(overlap(train_real['image_id'], test_df['image_id']))}",
    )
    add_check(
        "no_image_id_overlap_val_test",
        len(overlap(val_df["image_id"], test_df["image_id"])) == 0,
        f"overlap={len(overlap(val_df['image_id'], test_df['image_id']))}",
    )

    add_check(
        "no_group_id_overlap_train_val",
        len(overlap(train_real["group_id"], val_df["group_id"])) == 0,
        f"overlap={len(overlap(train_real['group_id'], val_df['group_id']))}",
    )
    add_check(
        "no_group_id_overlap_train_test",
        len(overlap(train_real["group_id"], test_df["group_id"])) == 0,
        f"overlap={len(overlap(train_real['group_id'], test_df['group_id']))}",
    )
    add_check(
        "no_group_id_overlap_val_test",
        len(overlap(val_df["group_id"], test_df["group_id"])) == 0,
        f"overlap={len(overlap(val_df['group_id'], test_df['group_id']))}",
    )

    add_check(
        "synthetic_absent_from_val",
        int((val_df["origin"] == "synthetic").sum()) == 0,
        f"n_synthetic={int((val_df['origin'] == 'synthetic').sum())}",
    )
    add_check(
        "synthetic_absent_from_test",
        int((test_df["origin"] == "synthetic").sum()) == 0,
        f"n_synthetic={int((test_df['origin'] == 'synthetic').sum())}",
    )

    synth_train = train_aug[train_aug["origin"] == "synthetic"].copy()
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])
    leaked_parent_to_val = synth_train["parent_image_id"].isin(val_ids).sum()
    leaked_parent_to_test = synth_train["parent_image_id"].isin(test_ids).sum()

    add_check(
        "no_synthetic_parent_leak_into_val",
        int(leaked_parent_to_val) == 0,
        f"n_leaks={int(leaked_parent_to_val)}",
    )
    add_check(
        "no_synthetic_parent_leak_into_test",
        int(leaked_parent_to_test) == 0,
        f"n_leaks={int(leaked_parent_to_test)}",
    )

    add_check(
        "augmented_train_contains_all_real_train_ids",
        set(train_real["image_id"]).issubset(set(train_aug["image_id"])),
        f"missing={len(set(train_real['image_id']) - set(train_aug['image_id']))}",
    )

    return pd.DataFrame(rows)


def audit_exact_duplicate_files(release: CleanCamRelease) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    hash_to_ids: Dict[str, List[str]] = defaultdict(list)

    for _, row in release.metadata.iterrows():
        path = Path(row["absolute_path"])
        try:
            sha = compute_sha256(path)
        except Exception as e:
            rows.append(
                {"status": "error", "image_id": row["image_id"], "details": str(e)}
            )
            continue
        hash_to_ids[sha].append(str(row["image_id"]))

    duplicate_groups = {sha: ids for sha, ids in hash_to_ids.items() if len(ids) > 1}
    if not duplicate_groups:
        return pd.DataFrame(
            [
                {
                    "status": "ok",
                    "n_duplicate_sha_groups": 0,
                    "details": "No exact duplicate file contents found.",
                }
            ]
        )

    for sha, ids in duplicate_groups.items():
        rows.append(
            {
                "status": "duplicate_sha",
                "sha256": sha,
                "n_images": len(ids),
                "image_ids": "|".join(ids),
            }
        )
    return pd.DataFrame(rows)


def audit_perceptual_near_duplicates(
    split_a: pd.DataFrame,
    split_b: pd.DataFrame,
    threshold: int = 4,
    max_images_per_split: Optional[int] = None,
) -> pd.DataFrame:
    a = split_a.copy()
    b = split_b.copy()
    if max_images_per_split is not None:
        a = a.head(max_images_per_split)
        b = b.head(max_images_per_split)

    a_hashes = []
    for _, row in a.iterrows():
        try:
            a_hashes.append(
                (row["image_id"], compute_phash(Path(row["absolute_path"])))
            )
        except Exception:
            continue

    b_hashes = []
    for _, row in b.iterrows():
        try:
            b_hashes.append(
                (row["image_id"], compute_phash(Path(row["absolute_path"])))
            )
        except Exception:
            continue

    findings = []
    for a_id, a_hash in a_hashes:
        for b_id, b_hash in b_hashes:
            dist = hamming_distance(a_hash, b_hash)
            if dist <= threshold:
                findings.append(
                    {
                        "split_a_image_id": a_id,
                        "split_b_image_id": b_id,
                        "hamming_distance": dist,
                    }
                )

    return pd.DataFrame(findings)


def run_integrity_audit(
    release: CleanCamRelease,
    out: OutputManager,
    run_near_duplicate_audit: bool,
    near_duplicate_cap: Optional[int],
) -> None:
    integrity = audit_official_split_integrity(release)
    exact_dup = audit_exact_duplicate_files(release)

    save_csv(integrity, out.tables_root / "split_integrity_report.csv")
    save_csv(exact_dup, out.tables_root / "exact_duplicate_audit.csv")

    summary = {
        "all_integrity_checks_passed": bool(integrity["passed"].all()),
        "integrity_checks": integrity.to_dict(orient="records"),
        "exact_duplicate_rows": exact_dup.to_dict(orient="records"),
    }

    if run_near_duplicate_audit:
        train_real = release.official_splits["train_real_only"]
        val_df = release.official_splits["val"]
        test_df = release.official_splits["test"]

        near_train_val = audit_perceptual_near_duplicates(
            train_real, val_df, max_images_per_split=near_duplicate_cap
        )
        near_train_test = audit_perceptual_near_duplicates(
            train_real, test_df, max_images_per_split=near_duplicate_cap
        )
        near_val_test = audit_perceptual_near_duplicates(
            val_df, test_df, max_images_per_split=near_duplicate_cap
        )

        save_csv(near_train_val, out.tables_root / "near_duplicate_train_vs_val.csv")
        save_csv(near_train_test, out.tables_root / "near_duplicate_train_vs_test.csv")
        save_csv(near_val_test, out.tables_root / "near_duplicate_val_vs_test.csv")

        summary["near_duplicate_audit"] = {
            "threshold": 4,
            "train_vs_val_rows": int(len(near_train_val)),
            "train_vs_test_rows": int(len(near_train_test)),
            "val_vs_test_rows": int(len(near_val_test)),
            "cap_per_split": near_duplicate_cap,
        }

    save_json(summary, out.summaries_root / "integrity_summary.json")


# ============================================================
# Annotation agreement
# ============================================================


def compute_annotation_agreement(
    annotation_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    required_cols = {"image_id", "annotator", "label"}
    missing = required_cols - set(annotation_df.columns)
    if missing:
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")

    pivot = annotation_df.pivot_table(
        index="image_id", columns="annotator", values="label", aggfunc="first"
    )
    annotators = list(pivot.columns)

    pair_rows: List[Dict[str, object]] = []

    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a_name = annotators[i]
            b_name = annotators[j]
            pair = pivot[[a_name, b_name]].dropna()
            if len(pair) == 0:
                continue

            y1 = pair[a_name].astype(int).to_numpy()
            y2 = pair[b_name].astype(int).to_numpy()

            pair_rows.append(
                {
                    "annotator_a": a_name,
                    "annotator_b": b_name,
                    "n_images": int(len(pair)),
                    "cohen_kappa": float(cohen_kappa_score(y1, y2)),
                    "quadratic_weighted_kappa": float(
                        cohen_kappa_score(y1, y2, weights="quadratic")
                    ),
                    "raw_agreement": float(np.mean(y1 == y2)),
                }
            )

    pair_df = pd.DataFrame(pair_rows)

    confusion_df = pd.DataFrame()
    if len(annotators) >= 2:
        first_pair = pivot[[annotators[0], annotators[1]]].dropna()
        if len(first_pair) > 0:
            cm = confusion_matrix(
                first_pair[annotators[0]].astype(int),
                first_pair[annotators[1]].astype(int),
                labels=LABELS,
            )
            confusion_df = pd.DataFrame(
                cm,
                index=[f"a_true_L{l}" for l in LABELS],
                columns=[f"b_pred_L{l}" for l in LABELS],
            )

    adjudication_rate = float("nan")
    resolved_summary = {}
    if "resolved_label" in annotation_df.columns:
        merged = annotation_df[["image_id", "annotator", "label"]].copy()
        resolved = annotation_df[["image_id", "resolved_label"]].drop_duplicates()
        ann_counts = annotation_df.groupby("image_id")["label"].nunique()
        adjudication_rate = float((ann_counts > 1).mean())
        resolved_summary = {
            "n_images_with_resolved_label": int(len(resolved)),
            "adjudication_rate": adjudication_rate,
        }

    summary = {
        "n_annotators": int(annotation_df["annotator"].nunique()),
        "n_unique_images": int(annotation_df["image_id"].nunique()),
        "pairwise_rows": pair_df.to_dict(orient="records"),
        **resolved_summary,
    }

    return pair_df, confusion_df, summary


def run_annotation_agreement(annotation_csv: Path, out: OutputManager) -> None:
    annotation_df = pd.read_csv(annotation_csv)
    pair_df, confusion_df, summary = compute_annotation_agreement(annotation_df)

    save_csv(pair_df, out.tables_root / "annotation_agreement_pairs.csv")
    if len(confusion_df) > 0:
        save_csv(
            confusion_df.reset_index(),
            out.tables_root / "annotation_confusion_first_pair.csv",
        )
    save_json(summary, out.summaries_root / "annotation_agreement_summary.json")

    compact_rows = []
    if len(pair_df) > 0:
        compact_rows.append(
            {
                "n_annotators": summary["n_annotators"],
                "n_images": summary["n_unique_images"],
                "mean_cohen_kappa": pair_df["cohen_kappa"].mean(),
                "mean_quadratic_weighted_kappa": pair_df[
                    "quadratic_weighted_kappa"
                ].mean(),
                "mean_raw_agreement": pair_df["raw_agreement"].mean(),
                "adjudication_rate": summary.get("adjudication_rate", np.nan),
            }
        )
    compact_df = pd.DataFrame(compact_rows)
    save_csv(compact_df, out.tables_root / "annotation_agreement_summary.csv")


# ============================================================
# Synthetic subset characterization
# ============================================================


def grayscale_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    hist = hist / max(hist.sum(), 1.0)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return float(-(hist * np.log2(hist)).sum())


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def rms_contrast(gray: np.ndarray) -> float:
    return float(gray.astype(np.float64).std())


def brightness_mean(gray: np.ndarray) -> float:
    return float(gray.astype(np.float64).mean())


def extract_low_level_stats(image_path: Path) -> Dict[str, float]:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Failed to read image for low-level stats: {image_path}")
    return {
        "laplacian_var": laplacian_variance(gray),
        "tenengrad": tenengrad(gray),
        "rms_contrast": rms_contrast(gray),
        "entropy": grayscale_entropy(gray),
        "brightness_mean": brightness_mean(gray),
    }


def summarize_synthetic_subset(
    release: CleanCamRelease, out: OutputManager, max_sample_per_group: Optional[int]
) -> None:
    synth = release.metadata_synth.copy()
    real = release.metadata_real.copy()

    if len(synth) == 0:
        save_json(
            {"message": "No synthetic subset found."},
            out.summaries_root / "synthetic_subset_summary.json",
        )
        return

    numeric_cols = [
        "rotation_deg",
        "coverage_scale",
        "blur_scale_factor",
        "opacity",
        "avg_blockage",
        "patch_count",
        "spatial_coverage",
        "effective_alpha_mean",
        "effective_alpha_p95",
        "covered_alpha_mean",
        "covered_alpha_p95",
    ]
    stats_rows = []
    for col in numeric_cols:
        if col in synth.columns:
            vals = pd.to_numeric(synth[col], errors="coerce").dropna()
            if len(vals) > 0:
                stats_rows.append(
                    {
                        "field": col,
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=0)),
                        "min": float(vals.min()),
                        "median": float(vals.median()),
                        "max": float(vals.max()),
                    }
                )
    param_summary_df = pd.DataFrame(stats_rows)
    save_csv(
        param_summary_df, out.tables_root / "synthetic_generation_parameter_summary.csv"
    )

    real_severe = real[real["label"].isin([4, 5])].copy()
    synth_severe = synth[synth["label"].isin([4, 5])].copy()

    if max_sample_per_group is not None:
        real_severe = real_severe.head(max_sample_per_group)
        synth_severe = synth_severe.head(max_sample_per_group)

    def collect_stats(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            try:
                stats = extract_low_level_stats(Path(row["absolute_path"]))
                stats["group"] = group_name
                stats["image_id"] = row["image_id"]
                stats["label"] = int(row["label"])
                rows.append(stats)
            except Exception:
                continue
        return pd.DataFrame(rows)

    real_stats = collect_stats(real_severe, "real_severe")
    synth_stats = collect_stats(synth_severe, "synthetic_severe")
    combined_stats = pd.concat([real_stats, synth_stats], ignore_index=True)

    save_csv(combined_stats, out.tables_root / "synthetic_vs_real_low_level_stats.csv")

    summary_rows = []
    for metric in [
        "laplacian_var",
        "tenengrad",
        "rms_contrast",
        "entropy",
        "brightness_mean",
    ]:
        row = {"metric": metric}
        for group_name, group_df in combined_stats.groupby("group"):
            vals = group_df[metric].dropna()
            if len(vals) == 0:
                continue
            row[f"{group_name}_mean"] = float(vals.mean())
            row[f"{group_name}_std"] = float(vals.std(ddof=0))
        if (
            wasserstein_distance is not None
            and len(real_stats) > 0
            and len(synth_stats) > 0
        ):
            row["wasserstein_real_vs_synth"] = float(
                wasserstein_distance(
                    real_stats[metric].dropna().to_numpy(),
                    synth_stats[metric].dropna().to_numpy(),
                )
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, out.tables_root / "synthetic_stats_summary.csv")

    metrics = ["laplacian_var", "tenengrad", "rms_contrast", "entropy"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        if len(real_stats) > 0:
            ax.hist(
                real_stats[metric].dropna(),
                bins=40,
                alpha=0.6,
                label="Real severe",
                density=True,
            )
        if len(synth_stats) > 0:
            ax.hist(
                synth_stats[metric].dropna(),
                bins=40,
                alpha=0.6,
                label="Synthetic severe",
                density=True,
            )
        ax.set_title(metric)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out.figures_root / "synthetic_real_stats_comparison.png", dpi=300)
    plt.close()

    save_json(
        {
            "n_real_severe_samples_used": int(len(real_stats)),
            "n_synthetic_severe_samples_used": int(len(synth_stats)),
            "summary_rows": summary_df.to_dict(orient="records"),
        },
        out.summaries_root / "synthetic_subset_summary.json",
    )


# ============================================================
# Benchmark dataset and transforms
# ============================================================


class CleanCamClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = image_to_pil(Path(row["absolute_path"]))
        x = self.transform(image)
        y = LABEL_TO_INDEX[int(row["label"])]
        return x, y, str(row["image_id"])


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def _replace_classifier_head(
    module: nn.Module, objective: str, num_classes: int = NUM_CLASSES
) -> nn.Module:
    if objective == "ce":
        return nn.Linear(module.in_features, num_classes)
    if objective == "coral":
        require_ordinal_dependencies(objective)
        return CoralLayer(size_in=module.in_features, num_classes=num_classes)
    if objective == "corn":
        require_ordinal_dependencies(objective)
        return nn.Linear(module.in_features, num_classes - 1)
    raise ValueError(f"Unsupported objective: {objective}")


def build_model(
    model_name: str, objective: str = "ce", num_classes: int = NUM_CLASSES
) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = _replace_classifier_head(
            model.fc, objective=objective, num_classes=num_classes
        )
        model.num_classes = num_classes
        model.objective = objective
        return model

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = _replace_classifier_head(
            model.classifier[1], objective=objective, num_classes=num_classes
        )
        model.num_classes = num_classes
        model.objective = objective
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = _replace_classifier_head(
            model.classifier[1], objective=objective, num_classes=num_classes
        )
        model.num_classes = num_classes
        model.objective = objective
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def _loader_kwargs(cfg: BenchmarkConfig) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return kwargs


def make_train_loader(
    df: pd.DataFrame, transform: transforms.Compose, cfg: BenchmarkConfig
) -> DataLoader:
    dataset = CleanCamClassificationDataset(df, transform)
    loader_kwargs = _loader_kwargs(cfg)

    if cfg.use_weighted_sampler:
        label_indices = [LABEL_TO_INDEX[int(label)] for label in df["label"].tolist()]
        counts = Counter(label_indices)
        weights = [1.0 / counts[idx] for idx in label_indices]
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            **loader_kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        **loader_kwargs,
    )


def make_eval_loader(
    df: pd.DataFrame, transform: transforms.Compose, cfg: BenchmarkConfig
) -> DataLoader:
    dataset = CleanCamClassificationDataset(df, transform)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **_loader_kwargs(cfg),
    )


def build_loss_fn(
    train_df: pd.DataFrame, cfg: BenchmarkConfig, device: torch.device, objective: str
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if objective == "ce":
        if cfg.use_class_weights:
            y = train_df["label"].astype(int).to_numpy()
            class_weights = compute_class_weight(
                class_weight="balanced", classes=np.array(LABELS), y=y
            )
            class_weights = torch.tensor(
                class_weights, dtype=torch.float32, device=device
            )
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()

    require_ordinal_dependencies(objective)
    if cfg.use_class_weights:
        print(
            f"[Warning] use_class_weights=True is only applied to CE. "
            f"Ignoring class-weighted loss for objective={objective}.",
            flush=True,
        )

    if objective == "coral":

        def _coral_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            levels = levels_from_labelbatch(y, num_classes=NUM_CLASSES).type_as(logits)
            return coral_loss(logits, levels)

        return _coral_loss

    if objective == "corn":

        def _corn_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return corn_loss(logits, y, num_classes=NUM_CLASSES)

        return _corn_loss

    raise ValueError(f"Unsupported objective: {objective}")


def format_label_distribution(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    return " | ".join([f"L{label}:{int(counts.loc[label])}" for label in LABELS])


def print_run_header(
    model_name: str,
    objective: str,
    setting_name: str,
    seed: int,
    device: torch.device,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: BenchmarkConfig,
) -> None:
    print("=" * 100, flush=True)
    print(
        f"[RunStart] model={model_name} objective={objective} setting={setting_name} seed={seed} "
        f"device={device} amp={cfg.use_amp and device.type == 'cuda'} "
        f"batch_size={cfg.batch_size} workers={cfg.num_workers} lr={cfg.learning_rate}",
        flush=True,
    )
    print(
        f"[Data] train={len(train_df)} ({format_label_distribution(train_df)}) | "
        f"val={len(val_df)} ({format_label_distribution(val_df)}) | "
        f"test={len(test_df)} ({format_label_distribution(test_df)})",
        flush=True,
    )


def wandb_log_if_available(
    run, payload: Dict[str, object], step: Optional[int] = None
) -> None:
    if run is not None:
        run.log(payload, step=step)


def maybe_init_wandb(
    cfg: BenchmarkConfig,
    model_name: str,
    objective: str,
    setting_name: str,
    seed: int,
    output_dir: Path,
):
    if not cfg.use_wandb:
        return None
    if wandb is None:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` or disable --use-wandb."
        )
    run_name = (
        f"{cfg.wandb_run_prefix}-{model_name}-{objective}-{setting_name}-seed{seed}"
    )
    group_name = f"{model_name}-{objective}-{setting_name}"
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
        name=run_name,
        group=group_name,
        dir=str(output_dir),
        config=asdict(cfg),
        reinit=True,
    )


def ordinal_cumulative_probs_to_class_probs(
    cumulative_probs: torch.Tensor,
) -> torch.Tensor:
    if cumulative_probs.ndim != 2:
        raise ValueError(
            f"Expected cumulative_probs to have shape [N, K-1], got {tuple(cumulative_probs.shape)}"
        )
    n_examples = cumulative_probs.shape[0]
    class_probs = cumulative_probs.new_zeros((n_examples, NUM_CLASSES))
    class_probs[:, 0] = 1.0 - cumulative_probs[:, 0]
    for idx in range(1, NUM_CLASSES - 1):
        class_probs[:, idx] = cumulative_probs[:, idx - 1] - cumulative_probs[:, idx]
    class_probs[:, NUM_CLASSES - 1] = cumulative_probs[:, NUM_CLASSES - 2]
    class_probs = torch.clamp(class_probs, min=0.0, max=1.0)
    normalizer = class_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    class_probs = class_probs / normalizer
    return class_probs


def logits_to_class_probs_and_preds(
    logits: torch.Tensor, objective: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if objective == "ce":
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        return probs, preds

    require_ordinal_dependencies(objective)

    if objective == "coral":
        cumulative_probs = torch.sigmoid(logits)
        probs = ordinal_cumulative_probs_to_class_probs(cumulative_probs)
        preds = proba_to_label(cumulative_probs).view(-1).long()
        preds = preds.clamp(min=0, max=NUM_CLASSES - 1)
        return probs, preds

    if objective == "corn":
        conditional_probs = torch.sigmoid(logits)
        cumulative_probs = torch.cumprod(conditional_probs, dim=1)
        probs = ordinal_cumulative_probs_to_class_probs(cumulative_probs)
        preds = corn_label_from_logits(logits).view(-1).long()
        preds = preds.clamp(min=0, max=NUM_CLASSES - 1)
        return probs, preds

    raise ValueError(f"Unsupported objective: {objective}")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    objective: str,
    criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, object]:
    model.eval()
    all_logits = []
    all_targets = []
    all_ids = []
    running_loss = 0.0
    n_seen = 0

    with torch.no_grad():
        for x, y, ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            if criterion is not None:
                loss = criterion(logits, y)
                running_loss += float(loss.item()) * x.shape[0]
                n_seen += x.shape[0]
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())
            all_ids.extend(list(ids))

    logits_t = torch.cat(all_logits, dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    probs_t, preds_t = logits_to_class_probs_and_preds(logits_t, objective=objective)

    logits = logits_t.numpy()
    targets = targets_t.numpy()
    probs = probs_t.numpy()
    preds = preds_t.numpy()

    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    per_class_rows = []
    for idx, label in INDEX_TO_LABEL.items():
        per_class_rows.append(
            {
                "label": label,
                "precision": float(per_class[0][idx]),
                "recall": float(per_class[1][idx]),
                "f1": float(per_class[2][idx]),
                "support": int(per_class[3][idx]),
            }
        )

    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = confusion_matrix(
        targets, preds, labels=list(range(NUM_CLASSES)), normalize="true"
    )

    true_labels = np.array([INDEX_TO_LABEL[t] for t in targets])
    pred_labels = np.array([INDEX_TO_LABEL[p] for p in preds])

    try:
        qwk = float(cohen_kappa_score(true_labels, pred_labels, weights="quadratic"))
    except Exception:
        qwk = float("nan")

    mae = float(np.mean(np.abs(true_labels - pred_labels)))
    within_1_accuracy = float(np.mean(np.abs(true_labels - pred_labels) <= 1))

    y_true_bin = np.isin(true_labels, [4, 5]).astype(int)
    y_pred_bin = np.isin(pred_labels, [4, 5]).astype(int)
    y_prob_bin = probs[:, LABEL_TO_INDEX[4]] + probs[:, LABEL_TO_INDEX[5]]

    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )

    try:
        binary_auroc = float(roc_auc_score(y_true_bin, y_prob_bin))
    except Exception:
        binary_auroc = float("nan")

    try:
        binary_auprc = float(average_precision_score(y_true_bin, y_prob_bin))
    except Exception:
        binary_auprc = float("nan")

    out = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "qwk": qwk,
        "mae": mae,
        "within_1_accuracy": within_1_accuracy,
        "per_class_rows": per_class_rows,
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm,
        "logits": logits,
        "probs": probs,
        "targets": targets,
        "preds": preds,
        "image_ids": all_ids,
        "binary_metrics": {
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "f1": float(binary_f1),
            "auroc": binary_auroc,
            "auprc": binary_auprc,
        },
    }
    if criterion is not None:
        out["loss"] = float(running_loss / max(n_seen, 1))
    return out


def train_one_setting(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    objective: str,
    setting_name: str,
    seed: int,
    cfg: BenchmarkConfig,
    output_dir: Path,
) -> Dict[str, object]:
    set_seed(seed)
    require_ordinal_dependencies(objective)

    device = torch.device(
        "cuda"
        if (cfg.train_on_gpu_if_available and torch.cuda.is_available())
        else "cpu"
    )
    ensure_dir(output_dir)

    train_tf, eval_tf = build_transforms(cfg.image_size)
    train_loader = make_train_loader(train_df, train_tf, cfg)
    val_loader = make_eval_loader(val_df, eval_tf, cfg)
    test_loader = make_eval_loader(test_df, eval_tf, cfg)

    print_run_header(
        model_name,
        objective,
        setting_name,
        seed,
        device,
        train_df,
        val_df,
        test_df,
        cfg,
    )

    run = maybe_init_wandb(cfg, model_name, objective, setting_name, seed, output_dir)

    model = build_model(model_name, objective=objective).to(device)
    criterion = build_loss_fn(train_df, cfg, device=device, objective=objective)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val_macro_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = -1

    train_log_rows = []
    overall_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_seen = 0

        for step, (x, y, _) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = (
                torch.cuda.amp.autocast
                if (cfg.use_amp and device.type == "cuda")
                else nullcontext
            )
            with autocast_ctx():
                logits = model(x)
                loss = criterion(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_size = x.shape[0]
            running_loss += float(loss.item()) * batch_size
            n_seen += batch_size

            if step % cfg.log_interval == 0 or step == len(train_loader):
                avg_loss_so_far = running_loss / max(n_seen, 1)
                print(
                    f"[Train] model={model_name} objective={objective} setting={setting_name} seed={seed} "
                    f"epoch={epoch}/{cfg.epochs} step={step}/{len(train_loader)} "
                    f"avg_loss={avg_loss_so_far:.4f}",
                    flush=True,
                )

        train_loss = running_loss / max(n_seen, 1)
        val_metrics = evaluate_model(
            model, val_loader, device, objective=objective, criterion=criterion
        )
        val_macro_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_qwk = val_metrics["qwk"]
        val_mae = val_metrics["mae"]
        val_within_1 = val_metrics["within_1_accuracy"]
        epoch_time = time.time() - epoch_start
        current_lr = float(optimizer.param_groups[0]["lr"])

        improved = val_macro_f1 > best_val_macro_f1
        scheduler.step(val_macro_f1)

        train_log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(val_metrics.get("loss", float("nan"))),
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_qwk": val_qwk,
                "val_mae": val_mae,
                "val_within_1_accuracy": val_within_1,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
                "improved": improved,
            }
        )

        wandb_log_if_available(
            run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": float(val_metrics.get("loss", float("nan"))),
                "val/accuracy": val_acc,
                "val/macro_f1": val_macro_f1,
                "val/qwk": val_qwk,
                "val/mae": val_mae,
                "val/within_1_accuracy": val_within_1,
                "train/lr": current_lr,
                "system/epoch_time_sec": epoch_time,
            },
            step=epoch,
        )

        if improved:
            best_val_macro_f1 = val_macro_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[EpochEnd] model={model_name} objective={objective} setting={setting_name} seed={seed} "
            f"epoch={epoch}/{cfg.epochs} train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} val_qwk={val_qwk:.4f} "
            f"val_mae={val_mae:.4f} val_within1={val_within_1:.4f} "
            f"lr={current_lr:.6g} time={epoch_time:.1f}s "
            f"{'BEST' if improved else f'patience={patience_counter}/{cfg.patience}'}",
            flush=True,
        )

        if patience_counter >= cfg.patience:
            print(
                f"[EarlyStop] model={model_name} objective={objective} setting={setting_name} seed={seed} "
                f"stopping at epoch={epoch} best_epoch={best_epoch} best_val_macro_f1={best_val_macro_f1:.4f}",
                flush=True,
            )
            break

    if best_state is None:
        if run is not None:
            run.finish()
        raise RuntimeError("Training failed to produce a best state.")

    model.load_state_dict(best_state)
    val_metrics = evaluate_model(
        model, val_loader, device, objective=objective, criterion=criterion
    )
    test_metrics = evaluate_model(
        model, test_loader, device, objective=objective, criterion=criterion
    )
    total_time = time.time() - overall_start

    save_csv(
        pd.DataFrame(train_log_rows),
        output_dir
        / f"train_log_{model_name}_{objective}_{setting_name}_seed{seed}.csv",
    )

    if cfg.save_best_checkpoints:
        torch.save(
            best_state,
            output_dir / f"best_{model_name}_{objective}_{setting_name}_seed{seed}.pt",
        )

    pred_rows = []
    for img_id, target_idx, pred_idx, prob_vec in zip(
        test_metrics["image_ids"],
        test_metrics["targets"],
        test_metrics["preds"],
        test_metrics["probs"],
    ):
        row = {
            "image_id": img_id,
            "target_label": INDEX_TO_LABEL[int(target_idx)],
            "pred_label": INDEX_TO_LABEL[int(pred_idx)],
        }
        for idx, label in INDEX_TO_LABEL.items():
            row[f"prob_L{label}"] = float(prob_vec[idx])
        pred_rows.append(row)
    save_csv(
        pd.DataFrame(pred_rows),
        output_dir
        / f"test_predictions_{model_name}_{objective}_{setting_name}_seed{seed}.csv",
    )

    print(
        f"[Test] model={model_name} objective={objective} setting={setting_name} seed={seed} "
        f"best_epoch={best_epoch} test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f} test_qwk={test_metrics['qwk']:.4f} "
        f"test_mae={test_metrics['mae']:.4f} test_within1={test_metrics['within_1_accuracy']:.4f} "
        f"total_time={total_time/60.0:.1f} min",
        flush=True,
    )

    test_payload = {
        "final/best_epoch": best_epoch,
        "final/total_time_sec": total_time,
        "test/accuracy": float(test_metrics["accuracy"]),
        "test/macro_f1": float(test_metrics["macro_f1"]),
        "test/qwk": float(test_metrics["qwk"]),
        "test/mae": float(test_metrics["mae"]),
        "test/within_1_accuracy": float(test_metrics["within_1_accuracy"]),
        "test/binary_precision": float(test_metrics["binary_metrics"]["precision"]),
        "test/binary_recall": float(test_metrics["binary_metrics"]["recall"]),
        "test/binary_f1": float(test_metrics["binary_metrics"]["f1"]),
        "test/binary_auroc": float(test_metrics["binary_metrics"]["auroc"]),
        "test/binary_auprc": float(test_metrics["binary_metrics"]["auprc"]),
    }
    for row in test_metrics["per_class_rows"]:
        label = row["label"]
        test_payload[f"test/L{label}_precision"] = float(row["precision"])
        test_payload[f"test/L{label}_recall"] = float(row["recall"])
        test_payload[f"test/L{label}_f1"] = float(row["f1"])

    wandb_log_if_available(
        run, test_payload, step=best_epoch if best_epoch > 0 else None
    )
    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_macro_f1"] = float(best_val_macro_f1)
        run.finish()

    return {
        "model_name": model_name,
        "objective": objective,
        "setting_name": setting_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_log_rows": train_log_rows,
        "total_time_sec": total_time,
    }


def plot_confusion_matrix(
    cm: np.ndarray, labels: Sequence[int], title: str, out_path: Path
) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, [f"L{l}" for l in labels])
    plt.yticks(ticks, [f"L{l}" for l in labels])
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=300)
    plt.close()


def aggregate_benchmark_results(
    results: List[Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_rows = []
    per_class_rows = []
    binary_rows = []

    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[
            (result["model_name"], result["objective"], result["setting_name"])
        ].append(result)

    for (model_name, objective, setting_name), group in grouped.items():
        accs = [g["test_metrics"]["accuracy"] for g in group]
        macro_f1s = [g["test_metrics"]["macro_f1"] for g in group]
        qwks = [g["test_metrics"]["qwk"] for g in group]
        maes = [g["test_metrics"]["mae"] for g in group]
        within1s = [g["test_metrics"]["within_1_accuracy"] for g in group]

        main_row = {
            "model": model_name,
            "objective": objective,
            "training_setting": setting_name,
            "n_seeds": len(group),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=0)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s, ddof=0)),
            "qwk_mean": float(np.mean(qwks)),
            "qwk_std": float(np.std(qwks, ddof=0)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes, ddof=0)),
            "within_1_accuracy_mean": float(np.mean(within1s)),
            "within_1_accuracy_std": float(np.std(within1s, ddof=0)),
        }

        for label in LABELS:
            metric_rows = []
            for g in group:
                mapping = {
                    row["label"]: row for row in g["test_metrics"]["per_class_rows"]
                }
                metric_rows.append(mapping[label])
            for metric_name in ["precision", "recall", "f1"]:
                vals = [row[metric_name] for row in metric_rows]
                main_row[f"L{label}_{metric_name}_mean"] = float(np.mean(vals))
                main_row[f"L{label}_{metric_name}_std"] = float(np.std(vals, ddof=0))

        main_rows.append(main_row)

        for label in LABELS:
            label_metrics = []
            for g in group:
                mapping = {
                    row["label"]: row for row in g["test_metrics"]["per_class_rows"]
                }
                label_metrics.append(mapping[label])

            per_class_rows.append(
                {
                    "model": model_name,
                    "objective": objective,
                    "training_setting": setting_name,
                    "label": label,
                    "precision_mean": float(
                        np.mean([m["precision"] for m in label_metrics])
                    ),
                    "precision_std": float(
                        np.std([m["precision"] for m in label_metrics], ddof=0)
                    ),
                    "recall_mean": float(np.mean([m["recall"] for m in label_metrics])),
                    "recall_std": float(
                        np.std([m["recall"] for m in label_metrics], ddof=0)
                    ),
                    "f1_mean": float(np.mean([m["f1"] for m in label_metrics])),
                    "f1_std": float(np.std([m["f1"] for m in label_metrics], ddof=0)),
                    "support_mean": float(
                        np.mean([m["support"] for m in label_metrics])
                    ),
                }
            )

        binary_metrics = [g["test_metrics"]["binary_metrics"] for g in group]
        binary_rows.append(
            {
                "model": model_name,
                "objective": objective,
                "training_setting": setting_name,
                "precision_mean": float(
                    np.mean([m["precision"] for m in binary_metrics])
                ),
                "precision_std": float(
                    np.std([m["precision"] for m in binary_metrics], ddof=0)
                ),
                "recall_mean": float(np.mean([m["recall"] for m in binary_metrics])),
                "recall_std": float(
                    np.std([m["recall"] for m in binary_metrics], ddof=0)
                ),
                "f1_mean": float(np.mean([m["f1"] for m in binary_metrics])),
                "f1_std": float(np.std([m["f1"] for m in binary_metrics], ddof=0)),
                "auroc_mean": float(np.mean([m["auroc"] for m in binary_metrics])),
                "auroc_std": float(
                    np.std([m["auroc"] for m in binary_metrics], ddof=0)
                ),
                "auprc_mean": float(np.mean([m["auprc"] for m in binary_metrics])),
                "auprc_std": float(
                    np.std([m["auprc"] for m in binary_metrics], ddof=0)
                ),
            }
        )

    return (
        pd.DataFrame(main_rows),
        pd.DataFrame(per_class_rows),
        pd.DataFrame(binary_rows),
    )


def make_setting_comparison_plot(
    main_df: pd.DataFrame, metric_col: str, out_path: Path, title: str
) -> None:
    if len(main_df) == 0:
        return

    plot_df = main_df.copy()
    include_objective = plot_df["objective"].nunique() > 1 or any(
        plot_df["objective"] != "ce"
    )
    if include_objective:
        plot_df["display_name"] = plot_df.apply(
            lambda row: f"{row['model']}\n{objective_display_name(row['objective'])}",
            axis=1,
        )
    else:
        plot_df["display_name"] = plot_df["model"]

    plt.figure(figsize=(max(10, 1.2 * len(plot_df["display_name"].unique())), 5))
    models_order = list(dict.fromkeys(plot_df["display_name"].tolist()))
    x = np.arange(len(models_order))
    width = 0.35

    real_only = plot_df[plot_df["training_setting"] == "real_only"].set_index(
        "display_name"
    )
    real_plus = plot_df[plot_df["training_setting"] == "real_plus_synthetic"].set_index(
        "display_name"
    )

    vals_real = [
        real_only.loc[m, metric_col] if m in real_only.index else np.nan
        for m in models_order
    ]
    vals_aug = [
        real_plus.loc[m, metric_col] if m in real_plus.index else np.nan
        for m in models_order
    ]

    plt.bar(x - width / 2, vals_real, width=width, label="Real-only")
    plt.bar(x + width / 2, vals_aug, width=width, label="Real+Synthetic")
    plt.xticks(x, models_order)
    plt.ylabel(metric_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_benchmarks(
    release: CleanCamRelease, out: OutputManager, cfg: BenchmarkConfig
) -> None:
    train_real_only = release.official_splits["train_real_only"].copy()
    train_real_plus_synth = release.official_splits["train_real_plus_synthetic"].copy()
    val_df = release.official_splits["val"].copy()
    test_df = release.official_splits["test"].copy()

    print(
        f"[BenchmarkSetup] train_real_only={len(train_real_only)} "
        f"train_real_plus_synthetic={len(train_real_plus_synth)} "
        f"val={len(val_df)} test={len(test_df)} objectives={list(cfg.objectives)}",
        flush=True,
    )

    run_results: List[Dict[str, object]] = []

    for model_name in cfg.models:
        for objective in cfg.objectives:
            for setting_name, train_df in [
                ("real_only", train_real_only),
                ("real_plus_synthetic", train_real_plus_synth),
            ]:
                for seed in cfg.seeds:
                    setting_dir = (
                        out.benchmark_root
                        / model_name
                        / objective
                        / setting_name
                        / f"seed_{seed}"
                    )
                    print(
                        f"[Benchmark] model={model_name} objective={objective} setting={setting_name} seed={seed}",
                        flush=True,
                    )
                    result = train_one_setting(
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        model_name=model_name,
                        objective=objective,
                        setting_name=setting_name,
                        seed=seed,
                        cfg=cfg,
                        output_dir=setting_dir,
                    )
                    run_results.append(result)

                    plot_confusion_matrix(
                        result["test_metrics"]["confusion_matrix_norm"],
                        LABELS,
                        title=f"{model_name} | {objective_display_name(objective)} | {setting_name} | seed={seed}",
                        out_path=setting_dir / "confusion_matrix_test_norm.png",
                    )

    main_df, per_class_df, binary_df = aggregate_benchmark_results(run_results)

    save_csv(main_df, out.tables_root / "benchmark_summary_main.csv")
    save_csv(per_class_df, out.tables_root / "benchmark_summary_per_class.csv")
    save_csv(binary_df, out.tables_root / "binary_operational_summary.csv")

    make_setting_comparison_plot(
        main_df,
        "macro_f1_mean",
        out.figures_root / "benchmark_macro_f1_comparison.png",
        title="Macro-F1: Real-only vs Real+Synthetic",
    )
    make_setting_comparison_plot(
        main_df,
        "qwk_mean",
        out.figures_root / "benchmark_qwk_comparison.png",
        title="QWK: Real-only vs Real+Synthetic",
    )
    make_setting_comparison_plot(
        main_df,
        "mae_mean",
        out.figures_root / "benchmark_mae_comparison.png",
        title="MAE: Real-only vs Real+Synthetic",
    )
    make_setting_comparison_plot(
        main_df,
        "L5_recall_mean",
        out.figures_root / "benchmark_L5_recall_comparison.png",
        title="Level-5 recall: Real-only vs Real+Synthetic",
    )
    make_setting_comparison_plot(
        main_df,
        "L4_recall_mean",
        out.figures_root / "benchmark_L4_recall_comparison.png",
        title="Level-4 recall: Real-only vs Real+Synthetic",
    )

    best_rows = (
        main_df.sort_values(
            ["model", "objective", "macro_f1_mean"], ascending=[True, True, False]
        )
        .groupby(["model", "objective"], sort=True)
        .head(1)
    )
    best_lookup = {
        (row["model"], row["objective"], row["training_setting"])
        for _, row in best_rows.iterrows()
    }

    for model_name, objective, setting_name in best_lookup:
        selected = [
            r
            for r in run_results
            if r["model_name"] == model_name
            and r["objective"] == objective
            and r["setting_name"] == setting_name
        ]
        cms = np.stack(
            [r["test_metrics"]["confusion_matrix_norm"] for r in selected], axis=0
        )
        mean_cm = cms.mean(axis=0)
        plot_confusion_matrix(
            mean_cm,
            LABELS,
            title=f"{model_name} | {objective_display_name(objective)} | {setting_name} | mean normalized confusion",
            out_path=out.figures_root
            / f"confusion_mean_{model_name}_{objective}_{setting_name}.png",
        )

    print("[BenchmarkDone] wrote benchmark tables, plots, and summaries", flush=True)

    save_json(
        {
            "benchmark_config": asdict(cfg),
            "main_rows": main_df.to_dict(orient="records"),
            "binary_rows": binary_df.to_dict(orient="records"),
        },
        out.summaries_root / "benchmark_summary.json",
    )


# ============================================================
# CLI orchestration
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CleanCam paper analysis and benchmark pipeline"
    )

    parser.add_argument(
        "--release-root", type=str, required=True, help="Path to CleanCam_release"
    )
    parser.add_argument(
        "--output-root", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--annotation-csv",
        type=str,
        default=None,
        help="Optional annotation agreement CSV",
    )

    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-characterization", action="store_true")
    parser.add_argument("--run-integrity", action="store_true")
    parser.add_argument("--run-annotation", action="store_true")
    parser.add_argument("--run-synthetic-stats", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")

    parser.add_argument("--run-near-duplicate-audit", action="store_true")
    parser.add_argument(
        "--near-duplicate-cap",
        type=int,
        default=1000,
        help="Cap images per split for pHash audit",
    )

    parser.add_argument(
        "--synthetic-stat-cap",
        type=int,
        default=2000,
        help="Max real severe and synthetic severe samples",
    )

    parser.add_argument(
        "--models", nargs="+", default=["mobilenet_v2", "resnet18", "efficientnet_b0"]
    )
    parser.add_argument(
        "--objectives", nargs="+", default=["ce"], choices=list(OBJECTIVES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--disable-weighted-sampler", action="store_true")
    parser.add_argument("--enable-class-weights", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cleancam-benchmarks")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-run-prefix", type=str, default="cleancam")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    release = CleanCamRelease(Path(args.release_root))
    out = OutputManager(Path(args.output_root))

    if args.run_all:
        args.run_characterization = True
        args.run_integrity = True
        args.run_synthetic_stats = True
        args.run_benchmark = True
        if args.annotation_csv is not None:
            args.run_annotation = True

    bench_cfg = BenchmarkConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seeds=tuple(args.seeds),
        models=tuple(args.models),
        objectives=tuple(args.objectives),
        use_weighted_sampler=not args.disable_weighted_sampler,
        use_class_weights=args.enable_class_weights,
        train_on_gpu_if_available=not args.cpu_only,
        save_best_checkpoints=not args.no_save_checkpoints,
        use_amp=not args.no_amp,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_run_prefix=args.wandb_run_prefix,
    )

    if bench_cfg.use_wandb and bench_cfg.wandb_mode == "disabled":
        print(
            "[Warning] --use-wandb was set but --wandb-mode=disabled. W&B logging will be skipped.",
            flush=True,
        )
        bench_cfg.use_wandb = False

    run_log = {
        "release_root": str(Path(args.release_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "benchmark_config": asdict(bench_cfg),
    }
    save_json(run_log, out.logs_root / "run_config.json")

    if args.run_characterization:
        print("[Run] characterization")
        run_characterization(release, out)

    if args.run_integrity:
        print("[Run] integrity audit")
        run_integrity_audit(
            release,
            out,
            run_near_duplicate_audit=args.run_near_duplicate_audit,
            near_duplicate_cap=args.near_duplicate_cap,
        )

    if args.run_annotation:
        if args.annotation_csv is None:
            raise ValueError("--run-annotation requires --annotation-csv")
        print("[Run] annotation agreement")
        run_annotation_agreement(Path(args.annotation_csv), out)

    if args.run_synthetic_stats:
        print("[Run] synthetic subset characterization")
        summarize_synthetic_subset(
            release, out, max_sample_per_group=args.synthetic_stat_cap
        )

    if args.run_benchmark:
        print("[Run] benchmarks")
        run_benchmarks(release, out, bench_cfg)

    print("Done.")


if __name__ == "__main__":
    main()

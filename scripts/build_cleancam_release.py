#!/usr/bin/env python3
"""
Build the CleanCam public release from curated real images and deposit assets.

The builder creates a release folder containing:

- real and synthetic images organised by label;
- master metadata and subset-specific metadata tables;
- official capture-disjoint train/validation/test split files;
- a dirt-asset manifest, split summary, and build summary;
- a README and a copy of this builder script.

The official split is deterministic at the capture level. Synthetic images are
created only from parents inside the same split, so parent information does not
cross train, validation, or test boundaries. All curated real-image filenames
must pass the release schema, and every derived image identifier must be unique.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================


REQUIRED_LABELS: Tuple[int, ...] = (1, 2, 3, 4, 5)
SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")


@dataclass
class BuildConfig:
    dataset_root: str = "raw/label_by_cam_artifacts"
    dirt_assets_dir: str = "raw/dirt_assets"
    release_root: str = "CleanCam_release"
    release_tag: str = "v2.0.0"
    seed: int = 42
    num_workers: int = max(1, (os.cpu_count() or 2) - 1)
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # Time block used only for grouping metadata.
    block_seconds: int = 600

    # Official capture-disjoint split used for the frozen release.
    official_test_capture_ids: Tuple[str, ...] = (
        "cambot2__morning__20250831",
        "cambot1__evening__20250913",
        "cambot2__evening__20250918",
        "cambot2__evening__20250911",
        "cambot1__morning__20250911",
        "cambot2__evening__20250916",
    )
    official_val_capture_ids: Tuple[str, ...] = (
        "cambot2__evening__20250919",
        "cambot2__morning__20250903",
        "cambot1__evening__20250907",
        "cambot2__morning__20250915",
        "cambot1__morning__20250912",
        "cambot1__evening__20250914",
    )
    protected_train_capture_ids: Tuple[str, ...] = (
        "cambot1__morning__20250917",
    )

    # Synthetic generation settings.
    target_synthetic_total: int = 3600
    synthetic_partition_ratios: Tuple[Tuple[str, float], ...] = (
        ("train", 2800 / 3600),
        ("val", 400 / 3600),
        ("test", 400 / 3600),
    )
    synthetic_target_ratios: Tuple[Tuple[int, float], ...] = (
        (3, 0.15),
        (4, 0.25),
        (5, 0.60),
    )
    synthetic_source_mix: Tuple[Tuple[int, Tuple[Tuple[int, float], ...]], ...] = (
        (3, ((1, 0.30), (2, 0.70))),
        (4, ((2, 0.35), (3, 0.65))),
        (5, ((2, 0.10), (3, 0.90))),
    )
    synthetic_asset_partition_ratios: Tuple[Tuple[str, float], ...] = (
        ("train", 5 / 11),
        ("val", 3 / 11),
        ("test", 3 / 11),
    )
    synthetic_max_per_parent_train: int = 2
    synthetic_max_per_parent_eval: int = 1
    synthetic_opacity_min: float = 0.10
    synthetic_opacity_max: float = 0.60
    synthetic_blur_min: int = 15
    synthetic_blur_max: int = 35
    synthetic_max_resized_width: int = 4000
    synthetic_max_attempt_factor: int = 20
    synthetic_save_jpeg_quality: int = 95

    # Release options.
    copy_dirt_assets_to_release: bool = True
    compute_sha256: bool = False

    def target_synthetic_by_label(self, total: Optional[int] = None) -> Dict[int, int]:
        ratios = dict(self.synthetic_target_ratios)
        if not ratios:
            raise ValueError("synthetic_target_ratios must not be empty")
        if any(lbl not in {3, 4, 5} for lbl in ratios):
            raise ValueError(f"synthetic target labels must be within {{3,4,5}}, got {sorted(ratios)}")
        if any(v <= 0 for v in ratios.values()):
            raise ValueError(f"synthetic target ratios must be positive, got {ratios}")

        target_total = self.target_synthetic_total if total is None else int(total)
        if target_total <= 0:
            raise ValueError(f"Synthetic target total must be positive, got {target_total}")

        return _allocate_integer_quotas_from_ratios(ratios, target_total)

    def target_synthetic_by_partition(self) -> Dict[str, int]:
        ratios = dict(self.synthetic_partition_ratios)
        if set(ratios) != set(SPLIT_NAMES):
            raise ValueError(
                f"synthetic_partition_ratios must specify exactly {SPLIT_NAMES}, got {sorted(ratios)}"
            )
        if any(v <= 0 for v in ratios.values()):
            raise ValueError(f"synthetic partition ratios must be positive, got {ratios}")
        return _allocate_integer_quotas_from_ratios(ratios, self.target_synthetic_total)

    def source_mix_for_target(self, target_label: int) -> Dict[int, float]:
        mix_map = {int(target): tuple(pairs) for target, pairs in self.synthetic_source_mix}
        if target_label not in mix_map:
            raise ValueError(f"No synthetic source mix configured for target label {target_label}")
        mix_pairs = mix_map[target_label]
        if any(src not in {1, 2, 3} for src, _ in mix_pairs):
            raise ValueError(f"Synthetic source labels must be within {{1,2,3}} for target {target_label}")
        if any(float(r) <= 0 for _, r in mix_pairs):
            raise ValueError(f"Synthetic source mix must be positive for target {target_label}")

        mix = {int(src): float(r) for src, r in mix_pairs}
        mix = {src: ratio / sum(mix.values()) for src, ratio in mix.items()}

        if target_label == 3:
            allowed = {1, 2}
        elif target_label == 4:
            allowed = {2, 3}
        elif target_label == 5:
            allowed = {2, 3}
        else:
            raise ValueError(f"Unsupported synthetic target label {target_label}")

        if set(mix) != allowed:
            raise ValueError(
                f"Synthetic source mix for target {target_label} must use exactly {sorted(allowed)}, got {sorted(mix)}"
            )
        if any(src >= target_label for src in mix):
            raise ValueError(f"Synthetic source labels must satisfy source < target for target {target_label}")
        if target_label == 3 and 3 in mix:
            raise ValueError("3 -> 3 synthetic generation is forbidden")
        if target_label == 5 and 1 in mix:
            raise ValueError("1 -> 5 synthetic generation is forbidden")
        return mix

    def source_priority_for_target(self, split_name: str, target_label: int) -> Tuple[int, ...]:
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Unknown split_name: {split_name}")

        mix = self.source_mix_for_target(target_label)
        priority = tuple(src for src, _ in sorted(mix.items(), key=lambda kv: (-kv[1], kv[0])))

        if target_label == 5 and split_name in {"val", "test"}:
            return (3,)
        return priority

    def max_per_parent_for_split(self, split_name: str) -> int:
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Unknown split_name: {split_name}")
        return self.synthetic_max_per_parent_train if split_name == "train" else self.synthetic_max_per_parent_eval


def _allocate_integer_quotas_from_ratios(ratios: Dict[object, float], total: int) -> Dict[object, int]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if not ratios:
        raise ValueError("ratios must not be empty")
    if any(v <= 0 for v in ratios.values()):
        raise ValueError(f"All ratios must be positive, got {ratios}")

    total_ratio = float(sum(ratios.values()))
    normalized = {k: float(v) / total_ratio for k, v in ratios.items()}
    raw = {k: total * normalized[k] for k in ratios}
    floor_quotas = {k: int(math.floor(raw[k])) for k in ratios}
    remainder = total - sum(floor_quotas.values())

    if remainder > 0:
        ranked = sorted(ratios.keys(), key=lambda k: (- (raw[k] - floor_quotas[k]), str(k)))
        for key in ranked[:remainder]:
            floor_quotas[key] += 1

    if sum(floor_quotas.values()) != total:
        raise RuntimeError("Quota allocation failed to conserve total count")
    return dict(floor_quotas)


# ============================================================
# Public metadata schema
# ============================================================

PUBLIC_METADATA_COLUMNS: List[str] = [
    "image_id",
    "origin",
    "synthetic",
    "label",
    "source_label",
    "target_label",
    "cam",
    "state",
    "cam_state",
    "day",
    "sec",
    "capture_id",
    "block_index",
    "group_id",
    "synthetic_split",
    "asset_split",
    "parent_image_id",
    "parent_label",
    "width",
    "height",
    "relative_path",
    "source_filename",
    "asset_id",
    "asset_filename",
    "rotation_deg",
    "coverage_scale",
    "blur_scale_factor",
    "opacity",
    "avg_blockage",
    "label_boost",
    "synthetic_seed",
    "generator_version",
    "sha256",
]

PUBLIC_METADATA_DTYPES: Dict[str, str] = {
    "image_id": "string",
    "origin": "string",
    "synthetic": "boolean",
    "label": "Int64",
    "source_label": "Int64",
    "target_label": "Int64",
    "cam": "string",
    "state": "string",
    "cam_state": "string",
    "day": "string",
    "sec": "Int64",
    "capture_id": "string",
    "block_index": "Int64",
    "group_id": "string",
    "synthetic_split": "string",
    "asset_split": "string",
    "parent_image_id": "string",
    "parent_label": "Int64",
    "width": "Int64",
    "height": "Int64",
    "relative_path": "string",
    "source_filename": "string",
    "asset_id": "string",
    "asset_filename": "string",
    "rotation_deg": "Int64",
    "coverage_scale": "Float64",
    "blur_scale_factor": "Int64",
    "opacity": "Float64",
    "avg_blockage": "Float64",
    "label_boost": "Int64",
    "synthetic_seed": "Int64",
    "generator_version": "string",
    "sha256": "string",
}


# ============================================================
# Naming and parsing
# ============================================================

_ORIGINAL_FILENAME_RE = re.compile(
    r"^(?P<prefix_date>\d{8})_"
    r"(?P<cam1>cambot\d+)_"
    r"(?P<capture_date1>\d{8})_"
    r"(?P<timecode1>090001|160001)_"
    r"(?P<cam2>cambot\d+)_"
    r"(?P<capture_date2>\d{8})_"
    r"(?P<timecode2>090001|160001)_"
    r"sec(?P<sec>\d+)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


def parse_metadata_from_filename(filename: str) -> Dict[str, object]:
    """
    Strict parser for original real-image filenames.

    Expected format:
    [prefix_date]_[cam]_[capture_date]_[timecode]_[cam]_[capture_date]_[timecode]_secX.jpg

    Publication rule:
    - prefix_date, capture_date1, capture_date2 must all be equal
    - cam1 and cam2 must match
    - timecode1 and timecode2 must match

    Invalid filenames cause the release build to stop.
    """
    name = os.path.basename(filename)
    m = _ORIGINAL_FILENAME_RE.match(name)

    if not m:
        return {
            "valid": False,
            "reason": "regex_no_match",
            "filename": name,
        }

    prefix_date = m.group("prefix_date")
    cam1 = m.group("cam1")
    capture_date1 = m.group("capture_date1")
    timecode1 = m.group("timecode1")
    cam2 = m.group("cam2")
    capture_date2 = m.group("capture_date2")
    timecode2 = m.group("timecode2")
    sec = int(m.group("sec"))

    if cam1 != cam2:
        return {
            "valid": False,
            "reason": "camera_mismatch",
            "filename": name,
            "prefix_date": prefix_date,
            "cam1": cam1,
            "cam2": cam2,
            "capture_date1": capture_date1,
            "capture_date2": capture_date2,
            "timecode1": timecode1,
            "timecode2": timecode2,
            "sec": sec,
        }

    if timecode1 != timecode2:
        return {
            "valid": False,
            "reason": "timecode_mismatch",
            "filename": name,
            "prefix_date": prefix_date,
            "cam1": cam1,
            "cam2": cam2,
            "capture_date1": capture_date1,
            "capture_date2": capture_date2,
            "timecode1": timecode1,
            "timecode2": timecode2,
            "sec": sec,
        }

    if not (prefix_date == capture_date1 == capture_date2):
        return {
            "valid": False,
            "reason": "date_mismatch_prefix_capture",
            "filename": name,
            "prefix_date": prefix_date,
            "cam1": cam1,
            "cam2": cam2,
            "capture_date1": capture_date1,
            "capture_date2": capture_date2,
            "timecode1": timecode1,
            "timecode2": timecode2,
            "sec": sec,
        }

    state = "morning" if timecode1 == "090001" else "evening"

    return {
        "valid": True,
        "reason": "ok",
        "filename": name,
        "prefix_date": prefix_date,
        "cam": cam1,
        "day": capture_date1,
        "state": state,
        "timecode": timecode1,
        "sec": sec,
        "capture_date1": capture_date1,
        "capture_date2": capture_date2,
    }


def make_group_id(cam: str, state: str, day: str, sec: int, block_seconds: int) -> Tuple[str, int, str]:
    block_index = -1 if sec < 0 else sec // block_seconds
    logical_capture_id = f"{cam}__{state}__{day}"
    group_id = f"{logical_capture_id}__blk{block_index:03d}"
    return group_id, block_index, logical_capture_id


def make_real_image_id(cam: str, state: str, day: str, sec: int) -> str:
    return f"R_{cam}_{state}_{day}_{sec:06d}"


def make_real_filename(image_id: str, label: int) -> str:
    return f"{image_id}_L{label}.jpg"


def make_synthetic_image_id(uid: int) -> str:
    return f"S_{uid:07d}"


def make_synthetic_filename(image_id: str, parent_id: str, src_label: int, dst_label: int) -> str:
    return f"{image_id}_src{src_label}_dst{dst_label}_parent-{parent_id}.jpg"


def safe_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


# ============================================================
# Utilities
# ============================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_image_shape(path: str) -> Tuple[int, int]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    h, w = img.shape[:2]
    return h, w


def count_by_label(df: pd.DataFrame) -> Dict[int, int]:
    if len(df) == 0:
        return {}
    counts = df["label"].value_counts().sort_index().to_dict()
    return {int(k): int(v) for k, v in counts.items()}


def parse_target_ratios(s: str) -> Tuple[Tuple[int, float], ...]:
    """
    Example: "3:0.15,4:0.25,5:0.60"
    """
    out: List[Tuple[int, float]] = []
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"Invalid target ratio piece: {piece}")
        lbl_str, ratio_str = piece.split(":", 1)
        lbl = int(lbl_str.strip())
        ratio = float(ratio_str.strip())
        out.append((lbl, ratio))
    if not out:
        raise ValueError("Parsed empty synthetic target ratio spec")
    return tuple(out)


def parse_named_ratios(s: str) -> Tuple[Tuple[str, float], ...]:
    """
    Example: "train:0.7777777778,val:0.1111111111,test:0.1111111111"
    """
    out: List[Tuple[str, float]] = []
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"Invalid named ratio piece: {piece}")
        name, ratio_str = piece.split(":", 1)
        out.append((name.strip(), float(ratio_str.strip())))
    if not out:
        raise ValueError("Parsed empty named ratio spec")
    return tuple(out)


def parse_source_mix(s: str) -> Tuple[Tuple[int, Tuple[Tuple[int, float], ...]], ...]:
    """
    Example: "3=1:0.30|2:0.70,4=2:0.35|3:0.65,5=2:0.10|3:0.90"
    """
    out: List[Tuple[int, Tuple[Tuple[int, float], ...]]] = []
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Invalid source mix piece: {piece}")
        target_str, rest = piece.split("=", 1)
        target_label = int(target_str.strip())
        src_pairs: List[Tuple[int, float]] = []
        for subpiece in rest.split("|"):
            subpiece = subpiece.strip()
            if not subpiece:
                continue
            if ":" not in subpiece:
                raise ValueError(f"Invalid source mix subpiece: {subpiece}")
            src_str, ratio_str = subpiece.split(":", 1)
            src_pairs.append((int(src_str.strip()), float(ratio_str.strip())))
        if not src_pairs:
            raise ValueError(f"Empty source mix for target label {target_label}")
        out.append((target_label, tuple(src_pairs)))
    if not out:
        raise ValueError("Parsed empty source mix spec")
    return tuple(out)


def sanitize_public_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    private_cols = [col for col in df.columns if col.startswith("_")]
    forbidden_cols = [
        "release_path",
        "source_dataset_path",
        "source_asset_path",
        "release_or_source_path",
        "source_path",
    ]
    drop_cols = [col for col in private_cols + forbidden_cols if col in df.columns]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def align_public_metadata_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = sanitize_public_dataframe(df).copy()
    for col in PUBLIC_METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[PUBLIC_METADATA_COLUMNS]
    for col, dtype in PUBLIC_METADATA_DTYPES.items():
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError):
            out[col] = out[col].astype("object")
    return out


def _validate_config(cfg: BuildConfig) -> None:
    _validate_official_split_config(cfg)

    partition_quotas = cfg.target_synthetic_by_partition()
    if sum(partition_quotas.values()) != cfg.target_synthetic_total:
        raise RuntimeError("Synthetic partition quotas do not sum to target_synthetic_total")

    for split_name in SPLIT_NAMES:
        _ = cfg.max_per_parent_for_split(split_name)

    for target_label in sorted(dict(cfg.synthetic_target_ratios)):
        cfg.source_mix_for_target(target_label)


# ============================================================
# Real data indexing and copying
# ============================================================


def index_and_copy_real_images(cfg: BuildConfig, release_root: Path) -> pd.DataFrame:
    dataset_root = Path(cfg.dataset_root)
    real_root = release_root / "images" / "real"
    ensure_dir(real_root)

    rows: List[Dict[str, object]] = []
    seen_image_ids: Set[str] = set()

    label_dirs = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not label_dirs:
        raise RuntimeError(f"No label directories found under {dataset_root}")

    print("Indexing and copying real images...")
    for label_dir in label_dirs:
        label = int(label_dir.name)
        dst_dir = real_root / f"label_{label}"
        ensure_dir(dst_dir)
        image_paths = sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in cfg.image_exts])

        for src_path in tqdm(image_paths, desc=f"real label {label}"):
            parsed = parse_metadata_from_filename(src_path.name)

            if not parsed["valid"]:
                raise RuntimeError(
                    "Real-image filename validation failed: "
                    f"label={label}, filename={src_path.name}, "
                    f"reason={parsed.get('reason', 'invalid_filename')}"
                )

            cam = str(parsed["cam"])
            state = str(parsed["state"])
            day = str(parsed["day"])
            sec = int(parsed["sec"])

            image_id = make_real_image_id(cam, state, day, sec)
            if image_id in seen_image_ids:
                raise RuntimeError(
                    "Real-image identifier validation failed: "
                    f"image_id={image_id}, label={label}, filename={src_path.name}"
                )

            seen_image_ids.add(image_id)

            dst_path = dst_dir / make_real_filename(image_id, label)
            shutil.copy2(src_path, dst_path)

            height, width = read_image_shape(str(src_path))
            group_id, block_index, capture_id = make_group_id(cam, state, day, sec, cfg.block_seconds)

            rows.append(
                {
                    "image_id": image_id,
                    "origin": "real",
                    "synthetic": False,
                    "label": label,
                    "source_label": label,
                    "target_label": label,
                    "cam": cam,
                    "state": state,
                    "cam_state": f"{cam}__{state}",
                    "day": day,
                    "sec": sec,
                    "capture_id": capture_id,
                    "block_index": block_index,
                    "group_id": group_id,
                    "synthetic_split": pd.NA,
                    "asset_split": pd.NA,
                    "parent_image_id": pd.NA,
                    "parent_label": pd.NA,
                    "width": width,
                    "height": height,
                    "relative_path": safe_relpath(dst_path, release_root),
                    "source_filename": src_path.name,
                    "asset_id": pd.NA,
                    "asset_filename": pd.NA,
                    "rotation_deg": pd.NA,
                    "coverage_scale": pd.NA,
                    "blur_scale_factor": pd.NA,
                    "opacity": pd.NA,
                    "avg_blockage": pd.NA,
                    "label_boost": pd.NA,
                    "synthetic_seed": pd.NA,
                    "generator_version": pd.NA,
                    "sha256": sha256_file(dst_path) if cfg.compute_sha256 else pd.NA,
                    "_source_dataset_path": str(src_path),
                }
            )

    real_df = pd.DataFrame(rows)
    real_df = real_df.sort_values(["label", "cam", "state", "day", "sec"]).reset_index(drop=True)

    print(f"Released real images: {len(real_df)}")
    return real_df


# ============================================================
# Dirt assets
# ============================================================


def build_dirt_asset_manifest(cfg: BuildConfig, release_root: Path) -> pd.DataFrame:
    assets_dir = Path(cfg.dirt_assets_dir)
    if not assets_dir.exists():
        raise RuntimeError(f"Dirt assets directory not found: {assets_dir}")

    asset_paths = sorted([p for p in assets_dir.iterdir() if p.is_file() and p.suffix.lower() in cfg.image_exts])
    if not asset_paths:
        raise RuntimeError(f"No dirt assets found in: {assets_dir}")

    assets_out_root = release_root / "assets" / "dirt_assets"
    if cfg.copy_dirt_assets_to_release:
        ensure_dir(assets_out_root)

    rows: List[Dict[str, object]] = []
    print("Preparing dirt asset manifest...")
    for idx, src_path in enumerate(tqdm(asset_paths, desc="assets"), start=1):
        asset_id = f"A_{idx:04d}"
        rel_path: object = pd.NA

        if cfg.copy_dirt_assets_to_release:
            copied_path = assets_out_root / f"{asset_id}{src_path.suffix.lower()}"
            shutil.copy2(src_path, copied_path)
            rel_path = safe_relpath(copied_path, release_root)

        rows.append(
            {
                "asset_id": asset_id,
                "asset_filename": src_path.name,
                "release_relative_path": rel_path,
                "_source_asset_path": str(src_path),
            }
        )

    return pd.DataFrame(rows)


def split_dirt_assets(asset_manifest_df: pd.DataFrame, cfg: BuildConfig) -> Dict[str, pd.DataFrame]:
    n_assets = len(asset_manifest_df)
    if n_assets == 0:
        raise RuntimeError("Dirt asset manifest is empty")

    if n_assets < 3:
        raise RuntimeError(
            f"At least 3 dirt assets are required for disjoint train/val/test asset pools, got {n_assets}"
        )

    ratios = dict(cfg.synthetic_asset_partition_ratios)
    if set(ratios) != set(SPLIT_NAMES):
        raise RuntimeError(
            f"synthetic_asset_partition_ratios must specify exactly {SPLIT_NAMES}, got {sorted(ratios)}"
        )

    counts = _allocate_integer_quotas_from_ratios(ratios, n_assets)
    for split_name in SPLIT_NAMES:
        if counts[split_name] <= 0:
            counts[split_name] = 1

    while sum(counts.values()) > n_assets:
        donor_candidates = [name for name in SPLIT_NAMES if counts[name] > 1]
        if not donor_candidates:
            raise RuntimeError("Unable to enforce non-empty disjoint asset pools")
        donor = max(donor_candidates, key=lambda name: (counts[name], name))
        counts[donor] -= 1

    while sum(counts.values()) < n_assets:
        receiver = min(SPLIT_NAMES, key=lambda name: (counts[name], name))
        counts[receiver] += 1

    if any(counts[name] <= 0 for name in SPLIT_NAMES):
        raise RuntimeError(f"Failed to assign at least one asset per split: {counts}")

    sorted_assets = asset_manifest_df.sort_values("asset_id").reset_index(drop=True)
    subsets: Dict[str, pd.DataFrame] = {}
    start = 0
    for split_name in SPLIT_NAMES:
        end = start + counts[split_name]
        subset = sorted_assets.iloc[start:end].copy().reset_index(drop=True)
        if subset.empty:
            raise RuntimeError(f"Asset subset for {split_name} is empty")
        subset["asset_split"] = split_name
        subsets[split_name] = subset
        start = end

    if start != n_assets:
        raise RuntimeError("Asset partition did not consume all assets")

    all_asset_ids = set(asset_manifest_df["asset_id"].astype(str).tolist())
    assigned_asset_ids: Set[str] = set()
    for split_name, subset in subsets.items():
        subset_ids = set(subset["asset_id"].astype(str).tolist())
        if assigned_asset_ids & subset_ids:
            raise RuntimeError(f"Asset leakage detected across synthetic asset pools for split {split_name}")
        assigned_asset_ids |= subset_ids
    if assigned_asset_ids != all_asset_ids:
        raise RuntimeError("Asset partition did not preserve all asset ids")

    return subsets


# ============================================================
# Synthetic generation
# ============================================================


_GLOBAL_DIRT_ASSETS: List[Tuple[str, str, np.ndarray]] = []
_GLOBAL_CFG: Optional[BuildConfig] = None


class ViewportFoulingBlender:
    def __init__(self, dirt_assets: List[Tuple[str, str, np.ndarray]], cfg: BuildConfig):
        self.dirt_assets = dirt_assets
        self.cfg = cfg

    def get_seamless_full_coverage(
        self,
        dirt_rgba: np.ndarray,
        target_w: int,
        target_h: int,
        py_rng: random.Random,
    ) -> Tuple[np.ndarray, float, int]:
        h, w = dirt_rgba.shape[:2]
        target_diag = math.sqrt(target_w ** 2 + target_h ** 2)
        scale = (target_diag / min(h, w)) * 1.1

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        if new_w > self.cfg.synthetic_max_resized_width:
            rescale = self.cfg.synthetic_max_resized_width / new_w
            new_w = max(1, int(new_w * rescale))
            new_h = max(1, int(new_h * rescale))

        dirt_resized = cv2.resize(dirt_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        angle = py_rng.randint(0, 360)
        M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
        dirt_rotated = cv2.warpAffine(
            dirt_resized,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        center_x, center_y = new_w // 2, new_h // 2
        start_x = max(0, center_x - (target_w // 2))
        start_y = max(0, center_y - (target_h // 2))

        dirt_crop = dirt_rotated[start_y:start_y + target_h, start_x:start_x + target_w]
        if dirt_crop.shape[0] != target_h or dirt_crop.shape[1] != target_w:
            dirt_crop = cv2.resize(dirt_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return dirt_crop, scale, angle

    def apply_downsample_blur(self, img_rgba: np.ndarray, py_rng: random.Random) -> Tuple[np.ndarray, int]:
        h, w = img_rgba.shape[:2]
        scale_factor = py_rng.randint(self.cfg.synthetic_blur_min, self.cfg.synthetic_blur_max)
        small_w = max(1, w // scale_factor)
        small_h = max(1, h // scale_factor)
        small = cv2.resize(img_rgba, (small_w, small_h), interpolation=cv2.INTER_AREA)
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return blurred, scale_factor

    @staticmethod
    def calculate_target_label(current_label: int, alpha_blurred: np.ndarray, opacity: float) -> Tuple[int, float, int]:
        avg_blockage = float(np.mean(alpha_blurred) / 255.0 * opacity)

        if avg_blockage < 0.02:
            boost = 0
        elif avg_blockage < 0.08:
            boost = 1
        elif avg_blockage < 0.15:
            boost = 2
        elif avg_blockage < 0.25:
            boost = 3
        else:
            boost = 4

        new_label = min(5, int(current_label) + boost)
        return new_label, avg_blockage, boost

    def overlay(self, bg_img: np.ndarray, current_label: int, seed: int) -> Dict[str, object]:
        if not self.dirt_assets:
            raise RuntimeError("No dirt assets loaded in worker")

        py_rng = random.Random(seed)
        asset_id, asset_filename, dirt_raw = self.dirt_assets[py_rng.randrange(len(self.dirt_assets))]

        bg_h, bg_w = bg_img.shape[:2]
        dirt_crop, coverage_scale, angle = self.get_seamless_full_coverage(dirt_raw, bg_w, bg_h, py_rng)
        dirt_blurred, blur_scale_factor = self.apply_downsample_blur(dirt_crop, py_rng)
        opacity = py_rng.uniform(self.cfg.synthetic_opacity_min, self.cfg.synthetic_opacity_max)

        b, g, r, a_blurred = cv2.split(dirt_blurred)
        target_label, avg_blockage, label_boost = self.calculate_target_label(current_label, a_blurred, opacity)
        if target_label == current_label:
            return {"accepted": False, "reason": "no_label_change"}

        bg_float = bg_img.astype(np.float32)
        dirt_float = cv2.merge((b, g, r)).astype(np.float32)
        alpha_float = (a_blurred.astype(np.float32) / 255.0) * opacity
        alpha_3ch = np.dstack([alpha_float] * 3)

        blended = bg_float + (dirt_float - bg_float) * alpha_3ch
        final_img = np.clip(blended, 0, 255).astype(np.uint8)

        return {
            "accepted": True,
            "final_img": final_img,
            "asset_id": asset_id,
            "asset_filename": asset_filename,
            "rotation_deg": int(angle),
            "coverage_scale": round(float(coverage_scale), 6),
            "blur_scale_factor": int(blur_scale_factor),
            "opacity": round(float(opacity), 6),
            "avg_blockage": round(float(avg_blockage), 6),
            "label_boost": int(label_boost),
            "target_label": int(target_label),
        }


def _load_rgba_asset(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read dirt asset: {path}")
    if img.ndim != 3:
        raise RuntimeError(f"Expected 3D image for asset: {path}")
    if img.shape[2] == 4:
        return img
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(img)
        return cv2.merge((b, g, r, alpha))
    raise RuntimeError(f"Unsupported channel count for asset: {path}")


def _worker_init(asset_manifest_rows: List[Dict[str, object]], cfg_dict: Dict[str, object]) -> None:
    global _GLOBAL_DIRT_ASSETS, _GLOBAL_CFG
    _GLOBAL_CFG = BuildConfig(**cfg_dict)
    loaded: List[Tuple[str, str, np.ndarray]] = []
    for row in asset_manifest_rows:
        loaded.append(
            (
                str(row["asset_id"]),
                str(row["asset_filename"]),
                _load_rgba_asset(str(row["_source_asset_path"])),
            )
        )
    _GLOBAL_DIRT_ASSETS = loaded


def _synthetic_task(task: Dict[str, object]) -> Dict[str, object]:
    global _GLOBAL_DIRT_ASSETS, _GLOBAL_CFG
    assert _GLOBAL_CFG is not None

    bg_img = cv2.imread(str(task["parent_source_path"]), cv2.IMREAD_COLOR)
    if bg_img is None:
        return {"accepted": False, "reason": f"failed_to_read_parent:{task['parent_source_path']}"}

    blender = ViewportFoulingBlender(_GLOBAL_DIRT_ASSETS, _GLOBAL_CFG)
    result = blender.overlay(bg_img=bg_img, current_label=int(task["source_label"]), seed=int(task["task_seed"]))
    if not result.get("accepted", False):
        return result

    if int(result["target_label"]) != int(task["desired_target_label"]):
        return {"accepted": False, "reason": f"wrong_target:{result['target_label']}"}

    release_path = Path(str(task["release_path"]))
    ensure_dir(release_path.parent)

    ok = cv2.imwrite(
        str(release_path),
        result["final_img"],
        [int(cv2.IMWRITE_JPEG_QUALITY), int(_GLOBAL_CFG.synthetic_save_jpeg_quality)],
    )
    if not ok:
        return {"accepted": False, "reason": f"failed_to_write:{release_path}"}

    height, width = result["final_img"].shape[:2]
    result.pop("final_img")
    result.update(
        {
            "accepted": True,
            "width": width,
            "height": height,
            "release_path": str(release_path),
        }
    )
    return result


def _rows_by_capture(pool_df: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for capture_id, subdf in pool_df.groupby("capture_id", sort=True):
        grouped[str(capture_id)] = subdf.sort_values("image_id").to_dict("records")
    return grouped


def _has_eligible_parent_in_grouped_rows(
    grouped_rows: Dict[str, List[Dict[str, object]]],
    parent_accept_counts: Dict[str, int],
    max_per_parent: int,
) -> bool:
    return any(
        parent_accept_counts[str(rec["image_id"])] < max_per_parent
        for records in grouped_rows.values()
        for rec in records
    )


def _choose_source_label_for_target(
    source_priority: Sequence[int],
    grouped_pools: Dict[int, Dict[str, List[Dict[str, object]]]],
    parent_accept_counts: Dict[str, int],
    max_per_parent: int,
) -> Optional[int]:
    for source_label in source_priority:
        grouped_rows = grouped_pools.get(source_label, {})
        if grouped_rows and _has_eligible_parent_in_grouped_rows(grouped_rows, parent_accept_counts, max_per_parent):
            return int(source_label)
    return None


def generate_synthetic_images_for_split(
    cfg: BuildConfig,
    release_root: Path,
    split_name: str,
    split_real_df: pd.DataFrame,
    split_asset_manifest_df: pd.DataFrame,
    target_total: int,
    image_uid_start: int,
) -> Tuple[pd.DataFrame, int]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unknown synthetic split name: {split_name}")
    if target_total <= 0:
        raise ValueError(f"Synthetic target total for split {split_name} must be positive, got {target_total}")
    if split_real_df.empty:
        raise RuntimeError(f"Real split {split_name} is empty; cannot generate synthetic children")
    if split_asset_manifest_df.empty:
        raise RuntimeError(f"Asset subset for split {split_name} is empty")

    synth_root = release_root / "images" / "synthetic"
    ensure_dir(synth_root)

    target_quotas = cfg.target_synthetic_by_label(total=target_total)
    source_priority_by_target: Dict[int, Tuple[int, ...]] = {}
    grouped_pools_by_target: Dict[int, Dict[int, Dict[str, List[Dict[str, object]]]]] = {}

    for target_label in sorted(target_quotas):
        configured_priority = cfg.source_priority_for_target(split_name, target_label)
        available_priority: List[int] = []
        grouped_pools: Dict[int, Dict[str, List[Dict[str, object]]]] = {}
        for source_label in configured_priority:
            source_pool = split_real_df.loc[split_real_df["label"] == source_label].copy().reset_index(drop=True)
            if source_pool.empty:
                continue
            grouped_rows = _rows_by_capture(source_pool)
            if grouped_rows:
                available_priority.append(int(source_label))
                grouped_pools[int(source_label)] = grouped_rows

        if not available_priority:
            raise RuntimeError(
                f"No eligible real parents found for split={split_name}, target_label={target_label}, "
                f"configured_source_priority={configured_priority}"
            )

        source_priority_by_target[target_label] = tuple(available_priority)
        grouped_pools_by_target[target_label] = grouped_pools

    parent_accept_counts: Dict[str, int] = Counter()
    max_per_parent = cfg.max_per_parent_for_split(split_name)
    accepted_rows: List[Dict[str, object]] = []
    accepted_by_target: Counter = Counter()
    accepted_by_target_source: Counter = Counter()
    attempts_by_target: Counter = Counter()
    global_uid = int(image_uid_start)

    manifest_rows = split_asset_manifest_df.to_dict("records")
    cfg_dict = asdict(cfg)
    batch_size = max(64, cfg.num_workers * 8)
    max_attempts = {
        target_label: max(quota * cfg.synthetic_max_attempt_factor, quota + 1)
        for target_label, quota in target_quotas.items()
    }
    samplers = {
        target_label: random.Random(
            cfg.seed + 100_000 * (1 + target_label) + {"train": 1, "val": 2, "test": 3}[split_name]
        )
        for target_label in target_quotas
    }

    source_priority_print = {target: list(priority) for target, priority in source_priority_by_target.items()}
    print(
        f"Generating synthetic images for split={split_name} with target total {target_total}, "
        f"label quotas {target_quotas}, source priority {source_priority_print}, "
        f"and asset ids {split_asset_manifest_df['asset_id'].astype(str).tolist()}"
    )

    with ProcessPoolExecutor(
        max_workers=cfg.num_workers,
        initializer=_worker_init,
        initargs=(manifest_rows, cfg_dict),
    ) as executor:
        progress = tqdm(total=sum(target_quotas.values()), desc=f"synthetic {split_name} accepted")
        try:
            for target_label in sorted(target_quotas):
                required = int(target_quotas[target_label])
                grouped_pools = grouped_pools_by_target[target_label]
                source_priority = source_priority_by_target[target_label]
                sampler = samplers[target_label]

                while accepted_by_target[target_label] < required:
                    if attempts_by_target[target_label] >= max_attempts[target_label]:
                        raise RuntimeError(
                            f"Reached max attempts for split={split_name}, target_label={target_label} before meeting quota. "
                            f"Accepted {accepted_by_target[target_label]}/{required}. "
                            f"Accepted-by-source={{"
                            + ", ".join(
                                f"{src}:{accepted_by_target_source[(target_label, src)]}" for src in source_priority
                            )
                            + "}}"
                        )

                    remaining = required - accepted_by_target[target_label]
                    current_batch = min(batch_size, max(remaining * 4, 32))
                    tasks: List[Dict[str, object]] = []

                    for _ in range(current_batch):
                        chosen_source = _choose_source_label_for_target(
                            source_priority,
                            grouped_pools,
                            parent_accept_counts,
                            max_per_parent,
                        )
                        if chosen_source is None:
                            if accepted_by_target[target_label] < required:
                                raise RuntimeError(
                                    f"Exhausted eligible parents for split={split_name}, target_label={target_label} "
                                    f"under max_per_parent={max_per_parent}. Accepted {accepted_by_target[target_label]}/{required}."
                                )
                            break

                        parent = _choose_parent_record(
                            grouped_pools[chosen_source],
                            parent_accept_counts,
                            max_per_parent,
                            sampler,
                        )
                        if parent is None:
                            continue

                        global_uid += 1
                        image_id = make_synthetic_image_id(global_uid)
                        filename = make_synthetic_filename(
                            image_id=image_id,
                            parent_id=str(parent["image_id"]),
                            src_label=int(parent["label"]),
                            dst_label=int(target_label),
                        )
                        release_path = synth_root / f"label_{target_label}" / filename

                        tasks.append(
                            {
                                "image_id": image_id,
                                "desired_target_label": int(target_label),
                                "source_label": int(parent["label"]),
                                "parent_image_id": str(parent["image_id"]),
                                "parent_label": int(parent["label"]),
                                "parent_source_path": str(parent["_source_dataset_path"]),
                                "parent_source_filename": str(parent["source_filename"]),
                                "cam": str(parent["cam"]),
                                "state": str(parent["state"]),
                                "cam_state": str(parent["cam_state"]),
                                "day": str(parent["day"]),
                                "sec": int(parent["sec"]),
                                "capture_id": str(parent["capture_id"]),
                                "block_index": int(parent["block_index"]),
                                "group_id": str(parent["group_id"]),
                                "synthetic_split": split_name,
                                "asset_split": split_name,
                                "release_path": str(release_path),
                                "task_seed": int(cfg.seed * 1_000_000 + global_uid),
                            }
                        )

                    if not tasks:
                        raise RuntimeError(
                            f"Could not schedule any synthetic tasks for split={split_name}, target_label={target_label}"
                        )

                    attempts_by_target[target_label] += len(tasks)

                    for task, result in zip(tasks, executor.map(_synthetic_task, tasks)):
                        if not result.get("accepted", False):
                            continue

                        if accepted_by_target[target_label] >= required:
                            try:
                                os.remove(result["release_path"])
                            except OSError:
                                pass
                            continue

                        parent_id = str(task["parent_image_id"])
                        if parent_accept_counts[parent_id] >= max_per_parent:
                            try:
                                os.remove(result["release_path"])
                            except OSError:
                                pass
                            continue

                        rel_path = safe_relpath(Path(result["release_path"]), release_root)

                        accepted_rows.append(
                            {
                                "image_id": task["image_id"],
                                "origin": "synthetic",
                                "synthetic": True,
                                "label": int(target_label),
                                "source_label": int(task["source_label"]),
                                "target_label": int(target_label),
                                "cam": task["cam"],
                                "state": task["state"],
                                "cam_state": task["cam_state"],
                                "day": task["day"],
                                "sec": int(task["sec"]),
                                "capture_id": task["capture_id"],
                                "block_index": int(task["block_index"]),
                                "group_id": task["group_id"],
                                "synthetic_split": task["synthetic_split"],
                                "asset_split": task["asset_split"],
                                "parent_image_id": parent_id,
                                "parent_label": int(task["parent_label"]),
                                "width": int(result["width"]),
                                "height": int(result["height"]),
                                "relative_path": rel_path,
                                "source_filename": task["parent_source_filename"],
                                "asset_id": result["asset_id"],
                                "asset_filename": result["asset_filename"],
                                "rotation_deg": int(result["rotation_deg"]),
                                "coverage_scale": float(result["coverage_scale"]),
                                "blur_scale_factor": int(result["blur_scale_factor"]),
                                "opacity": float(result["opacity"]),
                                "avg_blockage": float(result["avg_blockage"]),
                                "label_boost": int(result["label_boost"]),
                                "synthetic_seed": int(task["task_seed"]),
                                "generator_version": f"seamless_ultra_blur_{cfg.release_tag}",
                                "sha256": sha256_file(Path(result["release_path"])) if cfg.compute_sha256 else pd.NA,
                            }
                        )

                        accepted_by_target[target_label] += 1
                        accepted_by_target_source[(target_label, int(task["source_label"]))] += 1
                        parent_accept_counts[parent_id] += 1
                        progress.update(1)

                        if accepted_by_target[target_label] >= required:
                            break
        finally:
            progress.close()

    synth_df = pd.DataFrame(accepted_rows)
    if synth_df.empty:
        raise RuntimeError(f"Synthetic generation for split {split_name} produced an empty dataframe")

    expected_target_counts = cfg.target_synthetic_by_label(total=target_total)
    actual_target_counts = count_by_label(synth_df)
    if actual_target_counts != expected_target_counts:
        raise RuntimeError(
            f"Synthetic label quotas not met for split {split_name}. Expected={expected_target_counts}, Actual={actual_target_counts}"
        )

    if synth_df["synthetic_split"].nunique() != 1 or str(synth_df["synthetic_split"].iloc[0]) != split_name:
        raise RuntimeError(f"Synthetic split tagging error for split {split_name}")
    if synth_df["asset_split"].nunique() != 1 or str(synth_df["asset_split"].iloc[0]) != split_name:
        raise RuntimeError(f"Asset split tagging error for split {split_name}")

    allowed_assets = set(split_asset_manifest_df["asset_id"].astype(str).tolist())
    used_assets = set(synth_df["asset_id"].dropna().astype(str).tolist())
    if not used_assets <= allowed_assets:
        raise RuntimeError(
            f"Synthetic split {split_name} used asset ids outside its assigned asset pool: {sorted(used_assets - allowed_assets)}"
        )

    split_capture_ids = set(split_real_df["capture_id"].astype(str).unique().tolist())
    synth_capture_ids = set(synth_df["capture_id"].astype(str).unique().tolist())
    if not synth_capture_ids <= split_capture_ids:
        raise RuntimeError(
            f"Synthetic split {split_name} contains captures outside its parent real split: {sorted(synth_capture_ids - split_capture_ids)}"
        )

    synth_df = synth_df.sort_values(["label", "parent_image_id", "image_id"]).reset_index(drop=True)
    return synth_df, global_uid


def _choose_parent_record(
    grouped_rows: Dict[str, List[Dict[str, object]]],
    parent_accept_counts: Dict[str, int],
    max_per_parent: int,
    sampler: random.Random,
) -> Optional[Dict[str, object]]:
    """
    Choose one eligible real parent from a source-label pool.

    Strategy:
    - respect max_per_parent strictly
    - spread usage across captures first
    - then spread usage across parents within the chosen capture
    - keep tie-breaking deterministic through the provided sampler
    """
    eligible_by_capture: Dict[str, List[Dict[str, object]]] = {}

    for capture_id, records in grouped_rows.items():
        eligible_records = [
            rec
            for rec in records
            if parent_accept_counts[str(rec["image_id"])] < max_per_parent
        ]
        if eligible_records:
            eligible_by_capture[str(capture_id)] = eligible_records

    if not eligible_by_capture:
        return None

    # Balance parent usage across captures.
    capture_loads: Dict[str, int] = {
        capture_id: sum(parent_accept_counts[str(rec["image_id"])] for rec in records)
        for capture_id, records in eligible_by_capture.items()
    }
    min_capture_load = min(capture_loads.values())
    candidate_captures = sorted(
        [capture_id for capture_id, load in capture_loads.items() if load == min_capture_load]
    )
    chosen_capture = candidate_captures[sampler.randrange(len(candidate_captures))]

    # Then balance usage across parents within the chosen capture.
    eligible_records = eligible_by_capture[chosen_capture]
    min_parent_load = min(parent_accept_counts[str(rec["image_id"])] for rec in eligible_records)
    candidate_records = sorted(
        [
            rec
            for rec in eligible_records
            if parent_accept_counts[str(rec["image_id"])] == min_parent_load
        ],
        key=lambda rec: str(rec["image_id"]),
    )

    return candidate_records[sampler.randrange(len(candidate_records))]


def generate_all_synthetic_splits(
    cfg: BuildConfig,
    release_root: Path,
    official_real_splits: Dict[str, pd.DataFrame],
    asset_subsets: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    partition_totals = cfg.target_synthetic_by_partition()
    synthetic_splits: Dict[str, pd.DataFrame] = {}
    next_uid = 0

    for split_name in SPLIT_NAMES:
        synth_df, next_uid = generate_synthetic_images_for_split(
            cfg=cfg,
            release_root=release_root,
            split_name=split_name,
            split_real_df=official_real_splits[f"{split_name}_real_only"],
            split_asset_manifest_df=asset_subsets[split_name],
            target_total=partition_totals[split_name],
            image_uid_start=next_uid,
        )
        synthetic_splits[split_name] = synth_df

    all_synth = pd.concat(list(synthetic_splits.values()), ignore_index=True)
    if len(all_synth) != cfg.target_synthetic_total:
        raise RuntimeError(
            f"Total synthetic count mismatch. Expected {cfg.target_synthetic_total}, got {len(all_synth)}"
        )
    if len(all_synth["image_id"].astype(str).unique()) != len(all_synth):
        raise RuntimeError("Synthetic image_id collision detected across split-specific synthetic pools")

    return synthetic_splits


# ============================================================
# Split construction
# ============================================================


def _require_capture_ids_exist(real_df: pd.DataFrame, capture_ids: Set[str], split_name: str) -> None:
    existing_capture_ids = set(real_df["capture_id"].astype(str).unique().tolist())
    missing = sorted(capture_ids - existing_capture_ids)
    if missing:
        raise RuntimeError(f"{split_name} references capture_id values not found in real data: {missing}")


def _require_all_labels_present(df: pd.DataFrame, split_name: str) -> None:
    present = set(df["label"].dropna().astype(int).unique().tolist())
    missing = [label for label in REQUIRED_LABELS if label not in present]
    if missing:
        raise RuntimeError(f"{split_name} is missing required label(s): {missing}")


def _validate_official_split_config(cfg: BuildConfig) -> None:
    test_ids = set(cfg.official_test_capture_ids)
    val_ids = set(cfg.official_val_capture_ids)
    protected_train_ids = set(cfg.protected_train_capture_ids)

    if len(test_ids) != len(cfg.official_test_capture_ids):
        raise RuntimeError("Duplicate capture_id detected in official_test_capture_ids")
    if len(val_ids) != len(cfg.official_val_capture_ids):
        raise RuntimeError("Duplicate capture_id detected in official_val_capture_ids")
    if len(protected_train_ids) != len(cfg.protected_train_capture_ids):
        raise RuntimeError("Duplicate capture_id detected in protected_train_capture_ids")

    overlap_test_val = sorted(test_ids & val_ids)
    overlap_test_train = sorted(test_ids & protected_train_ids)
    overlap_val_train = sorted(val_ids & protected_train_ids)

    if overlap_test_val:
        raise RuntimeError(f"Official split config invalid: test and val overlap on capture_id: {overlap_test_val}")
    if overlap_test_train:
        raise RuntimeError(
            f"Official split config invalid: protected train capture(s) placed in test: {overlap_test_train}"
        )
    if overlap_val_train:
        raise RuntimeError(
            f"Official split config invalid: protected train capture(s) placed in val: {overlap_val_train}"
        )


def build_official_real_splits(real_df: pd.DataFrame, cfg: BuildConfig) -> Dict[str, pd.DataFrame]:
    _validate_official_split_config(cfg)

    test_capture_ids = set(cfg.official_test_capture_ids)
    val_capture_ids = set(cfg.official_val_capture_ids)
    protected_train_capture_ids = set(cfg.protected_train_capture_ids)

    _require_capture_ids_exist(real_df, test_capture_ids, "official_test_capture_ids")
    _require_capture_ids_exist(real_df, val_capture_ids, "official_val_capture_ids")
    _require_capture_ids_exist(real_df, protected_train_capture_ids, "protected_train_capture_ids")

    real_test = real_df.loc[real_df["capture_id"].isin(test_capture_ids)].copy()
    real_val = real_df.loc[real_df["capture_id"].isin(val_capture_ids)].copy()
    real_train = real_df.loc[
        ~real_df["capture_id"].isin(test_capture_ids | val_capture_ids)
    ].copy()

    if real_train.empty or real_val.empty or real_test.empty:
        raise RuntimeError("Official capture split produced an empty split")

    train_capture_ids = set(real_train["capture_id"].astype(str).unique().tolist())
    val_capture_ids_actual = set(real_val["capture_id"].astype(str).unique().tolist())
    test_capture_ids_actual = set(real_test["capture_id"].astype(str).unique().tolist())
    all_capture_ids = set(real_df["capture_id"].astype(str).unique().tolist())

    if train_capture_ids & val_capture_ids_actual:
        raise RuntimeError("Leakage detected: official train shares capture_id with validation")
    if train_capture_ids & test_capture_ids_actual:
        raise RuntimeError("Leakage detected: official train shares capture_id with test")
    if val_capture_ids_actual & test_capture_ids_actual:
        raise RuntimeError("Leakage detected: official validation shares capture_id with test")

    assigned_capture_ids = train_capture_ids | val_capture_ids_actual | test_capture_ids_actual
    if assigned_capture_ids != all_capture_ids:
        missing = sorted(all_capture_ids - assigned_capture_ids)
        extra = sorted(assigned_capture_ids - all_capture_ids)
        raise RuntimeError(
            f"Official split does not partition all captures cleanly. Missing={missing}, Extra={extra}"
        )

    missing_protected_train = sorted(protected_train_capture_ids - train_capture_ids)
    if missing_protected_train:
        raise RuntimeError(
            f"Protected training capture(s) are not in train_real_only: {missing_protected_train}"
        )

    _require_all_labels_present(real_train, "train_real_only")
    _require_all_labels_present(real_val, "val_real_only")
    _require_all_labels_present(real_test, "test_real_only")

    return {
        "train_real_only": real_train.sort_values("image_id").reset_index(drop=True),
        "val_real_only": real_val.sort_values("image_id").reset_index(drop=True),
        "test_real_only": real_test.sort_values("image_id").reset_index(drop=True),
    }


def assemble_official_splits(
    official_real_splits: Dict[str, pd.DataFrame],
    synthetic_splits: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    train_real_only = official_real_splits["train_real_only"].sort_values("image_id").reset_index(drop=True)
    val_real_only = official_real_splits["val_real_only"].sort_values("image_id").reset_index(drop=True)
    test_real_only = official_real_splits["test_real_only"].sort_values("image_id").reset_index(drop=True)

    val_synth = synthetic_splits["val"].sort_values("image_id").reset_index(drop=True)
    test_synth = synthetic_splits["test"].sort_values("image_id").reset_index(drop=True)
    train_synth = synthetic_splits["train"].sort_values("image_id").reset_index(drop=True)

    train_real_plus_synth = (
        pd.concat([train_real_only, train_synth], ignore_index=True)
        .sort_values(["origin", "label", "image_id"])
        .reset_index(drop=True)
    )
    val_real_plus_synth = (
        pd.concat([val_real_only, val_synth], ignore_index=True)
        .sort_values(["origin", "label", "image_id"])
        .reset_index(drop=True)
    )
    test_real_plus_synth = (
        pd.concat([test_real_only, test_synth], ignore_index=True)
        .sort_values(["origin", "label", "image_id"])
        .reset_index(drop=True)
    )

    return {
        "train_real_only": train_real_only,
        "train_real_plus_synthetic": train_real_plus_synth,
        "val_real_only": val_real_only,
        "val_real_plus_synthetic": val_real_plus_synth,
        "test_real_only": test_real_only,
        "test_real_plus_synthetic": test_real_plus_synth,
    }


# ============================================================
# Writing release artifacts
# ============================================================


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    sanitize_public_dataframe(df).to_csv(path, index=False)


def build_split_summary_rows(schema_name: str, fold_name: str, split_name: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    label_counts = df["label"].value_counts().reindex(list(REQUIRED_LABELS), fill_value=0)

    rows.append(
        {
            "schema": schema_name,
            "fold": fold_name,
            "split": split_name,
            "n_images": int(len(df)),
            "n_real": int((df["origin"] == "real").sum()),
            "n_synthetic": int((df["origin"] == "synthetic").sum()),
            "n_captures": int(df["capture_id"].nunique()),
            "n_groups": int(df["group_id"].nunique()),
            "label_1": int(label_counts.loc[1]),
            "label_2": int(label_counts.loc[2]),
            "label_3": int(label_counts.loc[3]),
            "label_4": int(label_counts.loc[4]),
            "label_5": int(label_counts.loc[5]),
        }
    )
    return rows


def write_release_readme(cfg: BuildConfig, release_root: Path) -> None:
    partition_totals = cfg.target_synthetic_by_partition()
    label_quotas_by_partition = {
        split_name: cfg.target_synthetic_by_label(total=partition_totals[split_name])
        for split_name in SPLIT_NAMES
    }
    readme = f"""# CleanCam release builder output

This release was generated by `build_cleancam_release.py` with release tag `{cfg.release_tag}`.

## Contents

- `images/real/`: all valid original field-captured images kept in the release.
- `images/synthetic/`: split-inherited synthetic samples generated by the viewport-fouling overlay pipeline.
- `metadata/metadata.csv`: master metadata for both real and synthetic images.
- `metadata/metadata_real.csv`: real-image subset.
- `metadata/metadata_synthetic.csv`: synthetic-image subset.
- `metadata/dirt_assets_manifest.csv`: dirt asset manifest.
- `metadata/split_summary.csv`: counts for the official recommended splits.
- `splits/official/`: official train/val/test split files for both real-only and real+synthetic benchmarks.

## Recommended evaluation protocol

The release exposes two official evaluation domains for both validation and test:

1. **Real-only benchmark**
   - Train with `train_real_only.csv` or `train_real_plus_synthetic.csv`
   - Select checkpoints on `val_real_only.csv`
   - Report on `test_real_only.csv`

2. **Mixed real+synthetic robustness benchmark**
   - Train with `train_real_only.csv` or `train_real_plus_synthetic.csv`
   - Select checkpoints on `val_real_plus_synthetic.csv`
   - Report on `test_real_plus_synthetic.csv`

This yields four official train/eval settings:
- real -> real
- real+synthetic -> real
- real -> real+synthetic
- real+synthetic -> real+synthetic

## Official split protocol

1. Index and validate all real images.
2. Partition the official real split at the `capture_id` level only.
3. Use the fixed explicit validation and test capture sets baked into this builder.
4. Keep configured protected capture(s) in training.
5. Generate synthetic samples separately inside each real split only:
   - train synthetic from `train_real_only`
   - val synthetic from `val_real_only`
   - test synthetic from `test_real_only`
6. Synthetic children inherit parent provenance and never cross parent splits.

## Leakage control

- Real train, val, and test are pairwise capture-disjoint.
- All groups and images from a capture remain in the same real split.
- Synthetic children stay in the same split as their real parent.
- Dirt assets are partitioned deterministically into split-specific subsets when enough assets are available.

## Synthetic defaults

- Total synthetic images: `{cfg.target_synthetic_total}`
- Synthetic partition totals: `{partition_totals}`
- Per-partition label quotas: `{label_quotas_by_partition}`
- Synthetic target ratios: `{dict(cfg.synthetic_target_ratios)}`
- Synthetic source priority:
  - train target `3`: `{cfg.source_priority_for_target('train', 3)}`
  - train target `4`: `{cfg.source_priority_for_target('train', 4)}`
  - train target `5`: `{cfg.source_priority_for_target('train', 5)}`
  - val target `5`: `{cfg.source_priority_for_target('val', 5)}`
  - test target `5`: `{cfg.source_priority_for_target('test', 5)}`
- `3 -> 3` is forbidden.
- `1 -> 5` is forbidden.

## Release integrity policy

Original real filenames are validated against the expected camera, date,
timecode, and elapsed-second schema. The build stops if an input violates that
schema or maps to a non-unique image identifier.
"""
    (release_root / "README.md").write_text(readme, encoding="utf-8")


def copy_builder_script(release_root: Path) -> None:
    dst = release_root / "code" / "build_cleancam_release.py"
    ensure_dir(dst.parent)
    shutil.copy2(Path(__file__), dst)


# ============================================================
# Main orchestration
# ============================================================


def write_metadata_outputs(
    release_root: Path,
    real_df: pd.DataFrame,
    synthetic_splits: Dict[str, pd.DataFrame],
    asset_manifest_df: pd.DataFrame,
    asset_subsets: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta_root = release_root / "metadata"
    ensure_dir(meta_root)

    synth_df = pd.concat(list(synthetic_splits.values()), ignore_index=True).sort_values("image_id").reset_index(drop=True)
    real_public = align_public_metadata_schema(real_df).sort_values("image_id").reset_index(drop=True)
    synth_public = align_public_metadata_schema(synth_df).sort_values("image_id").reset_index(drop=True)
    metadata_df = (
        pd.concat([real_public, synth_public], ignore_index=True)
        .sort_values(["origin", "label", "cam", "state", "day", "sec", "image_id"])
        .reset_index(drop=True)
    )

    asset_split_lookup = {
        str(row["asset_id"]): str(row["asset_split"])
        for split_name in SPLIT_NAMES
        for row in asset_subsets[split_name][["asset_id", "asset_split"]].to_dict("records")
    }
    asset_manifest_public = sanitize_public_dataframe(asset_manifest_df).copy()
    asset_manifest_public["asset_split"] = asset_manifest_public["asset_id"].astype(str).map(asset_split_lookup)

    write_dataframe(metadata_df, meta_root / "metadata.csv")
    write_dataframe(real_public, meta_root / "metadata_real.csv")
    write_dataframe(synth_public, meta_root / "metadata_synthetic.csv")
    write_dataframe(asset_manifest_public, meta_root / "dirt_assets_manifest.csv")

    return metadata_df, synth_public


def write_all_splits(
    release_root: Path,
    official_splits: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    split_root = release_root / "splits"
    ensure_dir(split_root)

    summary_rows: List[Dict[str, object]] = []

    official_root = split_root / "official"
    ensure_dir(official_root)
    for split_name, df in official_splits.items():
        write_dataframe(df, official_root / f"{split_name}.csv")
        summary_rows.extend(build_split_summary_rows("official", "official", split_name, sanitize_public_dataframe(df)))

    summary_df = pd.DataFrame(summary_rows)
    write_dataframe(summary_df, release_root / "metadata" / "split_summary.csv")
    return summary_df


def build_summary_json(
    cfg: BuildConfig,
    release_root: Path,
    real_df: pd.DataFrame,
    synthetic_splits: Dict[str, pd.DataFrame],
    metadata_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    asset_subsets: Dict[str, pd.DataFrame],
) -> None:
    synth_df = pd.concat(list(synthetic_splits.values()), ignore_index=True)
    out = {
        "release_tag": cfg.release_tag,
        "seed": cfg.seed,
        "block_seconds": cfg.block_seconds,
        "official_test_capture_ids": list(cfg.official_test_capture_ids),
        "official_val_capture_ids": list(cfg.official_val_capture_ids),
        "protected_train_capture_ids": list(cfg.protected_train_capture_ids),
        "real_count": int(len(real_df)),
        "synthetic_count": int(len(synth_df)),
        "total_count": int(len(metadata_df)),
        "real_label_counts": count_by_label(real_df),
        "synthetic_label_counts": count_by_label(synth_df) if len(synth_df) else {},
        "synthetic_split_counts": {split_name: int(len(df)) for split_name, df in synthetic_splits.items()},
        "synthetic_split_label_counts": {split_name: count_by_label(df) for split_name, df in synthetic_splits.items()},
        "synthetic_asset_ids": {
            split_name: asset_subsets[split_name]["asset_id"].astype(str).tolist() for split_name in SPLIT_NAMES
        },
        "all_label_counts": count_by_label(metadata_df),
        "cameras": sorted(metadata_df["cam"].dropna().astype(str).unique().tolist()) if len(metadata_df) else [],
        "states": sorted(metadata_df["state"].dropna().astype(str).unique().tolist()) if len(metadata_df) else [],
        "days": sorted(metadata_df["day"].dropna().astype(str).unique().tolist()) if len(metadata_df) else [],
        "official_rows": summary_df[summary_df["schema"] == "official"].to_dict("records"),
    }
    with (release_root / "metadata" / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def clean_existing_release_root(release_root: Path) -> None:
    if release_root.exists():
        shutil.rmtree(release_root)
    release_root.mkdir(parents=True, exist_ok=True)


def run(cfg: BuildConfig) -> None:
    _validate_config(cfg)
    seed_everything(cfg.seed)
    release_root = Path(cfg.release_root)

    clean_existing_release_root(release_root)
    real_df = index_and_copy_real_images(cfg, release_root)

    official_real_splits = build_official_real_splits(real_df, cfg)

    print("Official real split summary:")
    for split_name in ["train_real_only", "val_real_only", "test_real_only"]:
        split_df = official_real_splits[split_name]
        print(
            f"  {split_name}: n={len(split_df)} | "
            f"captures={split_df['capture_id'].nunique()} | "
            f"groups={split_df['group_id'].nunique()} | "
            f"labels={count_by_label(split_df)}"
        )

    asset_manifest_df = build_dirt_asset_manifest(cfg, release_root)
    asset_subsets = split_dirt_assets(asset_manifest_df, cfg)
    print("Synthetic asset pool summary:")
    for split_name in SPLIT_NAMES:
        subset = asset_subsets[split_name]
        print(f"  {split_name}: n_assets={len(subset)} | asset_ids={subset['asset_id'].astype(str).tolist()}")

    synthetic_splits = generate_all_synthetic_splits(
        cfg=cfg,
        release_root=release_root,
        official_real_splits=official_real_splits,
        asset_subsets=asset_subsets,
    )

    print("Synthetic split summary:")
    for split_name in SPLIT_NAMES:
        split_df = synthetic_splits[split_name]
        print(
            f"  synthetic_{split_name}: n={len(split_df)} | "
            f"captures={split_df['capture_id'].nunique()} | "
            f"groups={split_df['group_id'].nunique()} | "
            f"labels={count_by_label(split_df)}"
        )

    official_splits = assemble_official_splits(official_real_splits, synthetic_splits)
    metadata_df, synth_public = write_metadata_outputs(
        release_root,
        real_df,
        synthetic_splits,
        asset_manifest_df,
        asset_subsets,
    )
    split_summary_df = write_all_splits(release_root, official_splits)
    build_summary_json(
        cfg,
        release_root,
        real_df,
        synthetic_splits,
        metadata_df,
        split_summary_df,
        asset_subsets,
    )
    write_release_readme(cfg, release_root)
    copy_builder_script(release_root)

    print("\nBuild finished.")
    print(f"Release root: {release_root}")
    print(f"Released real images: {len(real_df)}")
    print(f"Released synthetic images: {len(synth_public)}")
    for split_name in [
        "train_real_only",
        "train_real_plus_synthetic",
        "val_real_only",
        "val_real_plus_synthetic",
        "test_real_only",
        "test_real_plus_synthetic",
    ]:
        split_df = official_splits[split_name]
        print(
            f"Official {split_name}: {len(split_df)} "
            f"(real={(split_df['origin'] == 'real').sum()}, synthetic={(split_df['origin'] == 'synthetic').sum()})"
        )
    print(f"Total released images in metadata.csv: {len(metadata_df)}")


# ============================================================
# CLI
# ============================================================


def parse_args() -> BuildConfig:
    defaults = BuildConfig()

    parser = argparse.ArgumentParser(description="Build the CleanCam release folder from curated real images and deposit assets")
    parser.add_argument("--dataset-root", default=defaults.dataset_root)
    parser.add_argument("--dirt-assets-dir", default=defaults.dirt_assets_dir)
    parser.add_argument("--release-root", default=defaults.release_root)
    parser.add_argument("--release-tag", default=defaults.release_tag)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--block-seconds", type=int, default=defaults.block_seconds)
    parser.add_argument("--target-synthetic-total", type=int, default=defaults.target_synthetic_total)
    parser.add_argument(
        "--synthetic-partition-ratios",
        type=str,
        default=",".join(f"{k}:{v}" for k, v in defaults.synthetic_partition_ratios),
        help='Example: "train:0.7777777778,val:0.1111111111,test:0.1111111111"',
    )
    parser.add_argument(
        "--synthetic-target-ratios",
        type=str,
        default=",".join(f"{k}:{v}" for k, v in defaults.synthetic_target_ratios),
        help='Example: "3:0.15,4:0.25,5:0.60"',
    )
    parser.add_argument(
        "--synthetic-source-mix",
        type=str,
        default=",".join(
            f"{target}=" + "|".join(f"{src}:{ratio}" for src, ratio in pairs)
            for target, pairs in defaults.synthetic_source_mix
        ),
        help='Example: "3=1:0.30|2:0.70,4=2:0.35|3:0.65,5=2:0.10|3:0.90"',
    )
    parser.add_argument(
        "--synthetic-asset-partition-ratios",
        type=str,
        default=",".join(f"{k}:{v}" for k, v in defaults.synthetic_asset_partition_ratios),
        help='Example: "train:0.4545454545,val:0.2727272727,test:0.2727272727"',
    )
    parser.add_argument("--copy-dirt-assets-to-release", action="store_true", default=defaults.copy_dirt_assets_to_release)
    parser.add_argument("--no-copy-dirt-assets-to-release", action="store_false", dest="copy_dirt_assets_to_release")
    parser.add_argument("--compute-sha256", action="store_true", default=defaults.compute_sha256)

    args = parser.parse_args()

    target_ratios = parse_target_ratios(args.synthetic_target_ratios)
    partition_ratios = parse_named_ratios(args.synthetic_partition_ratios)
    asset_partition_ratios = parse_named_ratios(args.synthetic_asset_partition_ratios)
    source_mix = parse_source_mix(args.synthetic_source_mix)

    return BuildConfig(
        dataset_root=args.dataset_root,
        dirt_assets_dir=args.dirt_assets_dir,
        release_root=args.release_root,
        release_tag=args.release_tag,
        seed=args.seed,
        num_workers=args.num_workers,
        block_seconds=args.block_seconds,
        official_test_capture_ids=defaults.official_test_capture_ids,
        official_val_capture_ids=defaults.official_val_capture_ids,
        protected_train_capture_ids=defaults.protected_train_capture_ids,
        target_synthetic_total=args.target_synthetic_total,
        synthetic_partition_ratios=partition_ratios,
        synthetic_target_ratios=target_ratios,
        synthetic_source_mix=source_mix,
        synthetic_asset_partition_ratios=asset_partition_ratios,
        copy_dirt_assets_to_release=args.copy_dirt_assets_to_release,
        compute_sha256=args.compute_sha256,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")
REQUIRED_LABELS: Tuple[int, ...] = (1, 2, 3, 4, 5)

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
]


@dataclass
class SyntheticConfig:
    dataset_root: str
    output_root: str
    seed: int = 42
    num_workers: int = max(1, (os.cpu_count() or 2) - 1)
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
    synthetic_max_per_parent_train: int = 2
    synthetic_max_per_parent_eval: int = 1
    synthetic_opacity_min: float = 0.10
    synthetic_opacity_max: float = 0.60
    synthetic_blur_min: int = 15
    synthetic_blur_max: int = 35
    synthetic_max_resized_width: int = 4000
    synthetic_max_attempt_factor: int = 20
    synthetic_save_jpeg_quality: int = 95
    generator_version: str = "seamless_ultra_blur_public"

    def target_synthetic_by_label(self, total: Optional[int] = None) -> Dict[int, int]:
        ratios = dict(self.synthetic_target_ratios)
        target_total = self.target_synthetic_total if total is None else int(total)
        return allocate_integer_quotas_from_ratios(ratios, target_total)

    def target_synthetic_by_partition(self) -> Dict[str, int]:
        return allocate_integer_quotas_from_ratios(dict(self.synthetic_partition_ratios), self.target_synthetic_total)

    def source_mix_for_target(self, target_label: int) -> Dict[int, float]:
        mix_map = {int(target): tuple(pairs) for target, pairs in self.synthetic_source_mix}
        if target_label not in mix_map:
            raise ValueError(f"Missing source mix for target label {target_label}")
        mix = {int(src): float(weight) for src, weight in mix_map[target_label]}
        total = sum(mix.values())
        mix = {src: weight / total for src, weight in mix.items()}

        if target_label == 3:
            allowed = {1, 2}
        elif target_label in {4, 5}:
            allowed = {2, 3}
        else:
            raise ValueError(f"Unsupported target label: {target_label}")

        if set(mix) != allowed:
            raise ValueError(f"Target {target_label} must use exactly source labels {sorted(allowed)}")
        if target_label == 3 and 3 in mix:
            raise ValueError("3 -> 3 generation is forbidden")
        if target_label == 5 and 1 in mix:
            raise ValueError("1 -> 5 generation is forbidden")
        return mix

    def source_priority_for_target(self, split_name: str, target_label: int) -> Tuple[int, ...]:
        mix = self.source_mix_for_target(target_label)
        priority = tuple(src for src, _ in sorted(mix.items(), key=lambda item: (-item[1], item[0])))
        if target_label == 5 and split_name in {"val", "test"}:
            return (3,)
        return priority

    def max_per_parent_for_split(self, split_name: str) -> int:
        return self.synthetic_max_per_parent_train if split_name == "train" else self.synthetic_max_per_parent_eval


def allocate_integer_quotas_from_ratios(ratios: Dict[object, float], total: int) -> Dict[object, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not ratios:
        raise ValueError("ratios must not be empty")
    if any(v <= 0 for v in ratios.values()):
        raise ValueError("all ratios must be positive")

    total_ratio = float(sum(ratios.values()))
    normalized = {k: float(v) / total_ratio for k, v in ratios.items()}
    raw = {k: total * normalized[k] for k in ratios}
    floor_quotas = {k: int(math.floor(raw[k])) for k in ratios}
    remainder = total - sum(floor_quotas.values())
    if remainder > 0:
        ranked = sorted(ratios.keys(), key=lambda k: (-(raw[k] - floor_quotas[k]), str(k)))
        for key in ranked[:remainder]:
            floor_quotas[key] += 1
    return dict(floor_quotas)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def count_by_label(df: pd.DataFrame) -> Dict[int, int]:
    if df.empty:
        return {}
    counts = df["label"].value_counts().sort_index().to_dict()
    return {int(k): int(v) for k, v in counts.items()}


def safe_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=drop_cols, errors="ignore")


def read_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)
    required = {"image_id", "label", "capture_id", "group_id", "relative_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split file {path} is missing columns: {sorted(missing)}")
    return df.copy()


def load_official_real_splits(dataset_root: Path) -> Dict[str, pd.DataFrame]:
    split_root = dataset_root / "splits" / "official"
    return {
        "train_real_only": read_split_csv(split_root / "train_real_only.csv"),
        "val_real_only": read_split_csv(split_root / "val_real_only.csv"),
        "test_real_only": read_split_csv(split_root / "test_real_only.csv"),
    }


def load_asset_manifest(dataset_root: Path) -> pd.DataFrame:
    manifest_path = dataset_root / "metadata" / "dirt_assets_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing asset manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)

    if "asset_split" not in df.columns:
        raise ValueError("metadata/dirt_assets_manifest.csv must contain an asset_split column")

    rel_col = None
    for candidate in ("release_relative_path", "relative_path"):
        if candidate in df.columns:
            rel_col = candidate
            break
    if rel_col is None:
        raise ValueError("Asset manifest must contain release_relative_path or relative_path")

    df = df.copy()
    df["asset_relative_path"] = df[rel_col]
    required = {"asset_id", "asset_filename", "asset_split", "asset_relative_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Asset manifest is missing columns: {sorted(missing)}")
    return df


_GLOBAL_DIRT_ASSETS: List[Tuple[str, str, np.ndarray]] = []
_GLOBAL_CFG: Optional[SyntheticConfig] = None


class SeamlessUltraBlurBlender:
    def __init__(self, dirt_assets: List[Tuple[str, str, np.ndarray]], cfg: SyntheticConfig):
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
        target_diag = math.sqrt(target_w**2 + target_h**2)
        scale = (target_diag / min(h, w)) * 1.1

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        if new_w > self.cfg.synthetic_max_resized_width:
            rescale = self.cfg.synthetic_max_resized_width / new_w
            new_w = max(1, int(new_w * rescale))
            new_h = max(1, int(new_h * rescale))

        dirt_resized = cv2.resize(dirt_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        angle = py_rng.randint(0, 360)
        matrix = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
        dirt_rotated = cv2.warpAffine(
            dirt_resized,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        center_x, center_y = new_w // 2, new_h // 2
        start_x = max(0, center_x - target_w // 2)
        start_y = max(0, center_y - target_h // 2)
        dirt_crop = dirt_rotated[start_y : start_y + target_h, start_x : start_x + target_w]
        if dirt_crop.shape[0] != target_h or dirt_crop.shape[1] != target_w:
            dirt_crop = cv2.resize(dirt_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return dirt_crop, scale, angle

    def apply_fake_blur(self, img_rgba: np.ndarray, py_rng: random.Random) -> Tuple[np.ndarray, int]:
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
            raise RuntimeError("No dirt assets loaded")

        py_rng = random.Random(seed)
        asset_id, asset_filename, dirt_raw = self.dirt_assets[py_rng.randrange(len(self.dirt_assets))]

        bg_h, bg_w = bg_img.shape[:2]
        dirt_crop, coverage_scale, angle = self.get_seamless_full_coverage(dirt_raw, bg_w, bg_h, py_rng)
        dirt_blurred, blur_scale_factor = self.apply_fake_blur(dirt_crop, py_rng)
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


def load_rgba_asset(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read dirt asset: {path}")
    if img.ndim != 3:
        raise RuntimeError(f"Unsupported asset shape: {path}")
    if img.shape[2] == 4:
        return img
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(img)
        return cv2.merge((b, g, r, alpha))
    raise RuntimeError(f"Unsupported channel count for asset: {path}")


def worker_init(asset_rows: List[Dict[str, object]], cfg_dict: Dict[str, object]) -> None:
    global _GLOBAL_CFG, _GLOBAL_DIRT_ASSETS
    _GLOBAL_CFG = SyntheticConfig(**cfg_dict)
    loaded: List[Tuple[str, str, np.ndarray]] = []
    for row in asset_rows:
        loaded.append(
            (
                str(row["asset_id"]),
                str(row["asset_filename"]),
                load_rgba_asset(str(row["_asset_abs_path"])),
            )
        )
    _GLOBAL_DIRT_ASSETS = loaded


def synthetic_task(task: Dict[str, object]) -> Dict[str, object]:
    global _GLOBAL_CFG, _GLOBAL_DIRT_ASSETS
    assert _GLOBAL_CFG is not None

    bg_img = cv2.imread(str(task["parent_abs_path"]), cv2.IMREAD_COLOR)
    if bg_img is None:
        return {"accepted": False, "reason": f"failed_to_read_parent:{task['parent_abs_path']}"}

    blender = SeamlessUltraBlurBlender(_GLOBAL_DIRT_ASSETS, _GLOBAL_CFG)
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
    result.update({"accepted": True, "width": width, "height": height, "release_path": str(release_path)})
    return result


def rows_by_capture(pool_df: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for capture_id, subdf in pool_df.groupby("capture_id", sort=True):
        grouped[str(capture_id)] = subdf.sort_values("image_id").to_dict("records")
    return grouped


def has_eligible_parent(grouped_rows: Dict[str, List[Dict[str, object]]], parent_accept_counts: Dict[str, int], max_per_parent: int) -> bool:
    return any(parent_accept_counts[str(rec["image_id"])] < max_per_parent for records in grouped_rows.values() for rec in records)


def choose_source_label(
    source_priority: Sequence[int],
    grouped_pools: Dict[int, Dict[str, List[Dict[str, object]]]],
    parent_accept_counts: Dict[str, int],
    max_per_parent: int,
) -> Optional[int]:
    for source_label in source_priority:
        grouped_rows = grouped_pools.get(source_label, {})
        if grouped_rows and has_eligible_parent(grouped_rows, parent_accept_counts, max_per_parent):
            return int(source_label)
    return None


def choose_parent_record(
    grouped_rows: Dict[str, List[Dict[str, object]]],
    parent_accept_counts: Dict[str, int],
    max_per_parent: int,
    sampler: random.Random,
) -> Optional[Dict[str, object]]:
    eligible_by_capture: Dict[str, List[Dict[str, object]]] = {}
    for capture_id, records in grouped_rows.items():
        eligible = [rec for rec in records if parent_accept_counts[str(rec["image_id"])] < max_per_parent]
        if eligible:
            eligible_by_capture[capture_id] = eligible
    if not eligible_by_capture:
        return None

    capture_loads = {
        capture_id: sum(parent_accept_counts[str(rec["image_id"])] for rec in records)
        for capture_id, records in eligible_by_capture.items()
    }
    min_capture_load = min(capture_loads.values())
    candidate_captures = sorted([capture_id for capture_id, load in capture_loads.items() if load == min_capture_load])
    chosen_capture = candidate_captures[sampler.randrange(len(candidate_captures))]

    eligible_records = eligible_by_capture[chosen_capture]
    min_parent_load = min(parent_accept_counts[str(rec["image_id"])] for rec in eligible_records)
    candidate_records = sorted(
        [rec for rec in eligible_records if parent_accept_counts[str(rec["image_id"])] == min_parent_load],
        key=lambda rec: str(rec["image_id"]),
    )
    return candidate_records[sampler.randrange(len(candidate_records))]


def make_synthetic_image_id(uid: int) -> str:
    return f"S_{uid:07d}"


def make_synthetic_filename(image_id: str, parent_id: str, src_label: int, dst_label: int) -> str:
    return f"{image_id}_src{src_label}_dst{dst_label}_parent-{parent_id}.jpg"


def prepare_split_df(df: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    out = df.copy()
    out["parent_abs_path"] = out["relative_path"].apply(lambda rel: str((dataset_root / str(rel)).resolve()))
    missing_paths = [p for p in out["parent_abs_path"].tolist() if not Path(p).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing parent image files under dataset root. Example: {missing_paths[0]}")
    return out


def prepare_asset_manifest(df: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    out = df.copy()
    out["_asset_abs_path"] = out["asset_relative_path"].apply(lambda rel: str((dataset_root / str(rel)).resolve()))
    missing_paths = [p for p in out["_asset_abs_path"].tolist() if not Path(p).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing dirt asset files under dataset root. Example: {missing_paths[0]}")
    return out


def generate_synthetic_images_for_split(
    cfg: SyntheticConfig,
    dataset_root: Path,
    output_root: Path,
    split_name: str,
    split_real_df: pd.DataFrame,
    split_asset_manifest_df: pd.DataFrame,
    target_total: int,
    image_uid_start: int,
) -> Tuple[pd.DataFrame, int]:
    split_real_df = prepare_split_df(split_real_df, dataset_root)
    split_asset_manifest_df = prepare_asset_manifest(split_asset_manifest_df, dataset_root)

    synth_root = output_root / "images" / "synthetic"
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
            grouped_rows = rows_by_capture(source_pool)
            if grouped_rows:
                available_priority.append(int(source_label))
                grouped_pools[int(source_label)] = grouped_rows
        if not available_priority:
            raise RuntimeError(
                f"No eligible parents found for split={split_name}, target_label={target_label}, configured_source_priority={configured_priority}"
            )
        source_priority_by_target[target_label] = tuple(available_priority)
        grouped_pools_by_target[target_label] = grouped_pools

    parent_accept_counts: Dict[str, int] = Counter()
    max_per_parent = cfg.max_per_parent_for_split(split_name)
    accepted_rows: List[Dict[str, object]] = []
    accepted_by_target: Counter = Counter()
    attempts_by_target: Counter = Counter()
    global_uid = int(image_uid_start)
    batch_size = max(64, cfg.num_workers * 8)
    max_attempts = {label: max(quota * cfg.synthetic_max_attempt_factor, quota + 1) for label, quota in target_quotas.items()}
    samplers = {
        label: random.Random(cfg.seed + 100_000 * (1 + label) + {"train": 1, "val": 2, "test": 3}[split_name])
        for label in target_quotas
    }

    manifest_rows = split_asset_manifest_df.to_dict("records")
    cfg_dict = asdict(cfg)

    with ProcessPoolExecutor(max_workers=cfg.num_workers, initializer=worker_init, initargs=(manifest_rows, cfg_dict)) as executor:
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
                            f"Accepted {accepted_by_target[target_label]}/{required}."
                        )

                    remaining = required - accepted_by_target[target_label]
                    current_batch = min(batch_size, max(remaining * 4, 32))
                    tasks: List[Dict[str, object]] = []

                    for _ in range(current_batch):
                        chosen_source = choose_source_label(source_priority, grouped_pools, parent_accept_counts, max_per_parent)
                        if chosen_source is None:
                            raise RuntimeError(
                                f"Exhausted eligible parents for split={split_name}, target_label={target_label}, max_per_parent={max_per_parent}."
                            )

                        parent = choose_parent_record(grouped_pools[chosen_source], parent_accept_counts, max_per_parent, sampler)
                        if parent is None:
                            continue

                        global_uid += 1
                        image_id = make_synthetic_image_id(global_uid)
                        filename = make_synthetic_filename(image_id, str(parent["image_id"]), int(parent["label"]), int(target_label))
                        release_path = synth_root / f"label_{target_label}" / filename

                        tasks.append(
                            {
                                "image_id": image_id,
                                "desired_target_label": int(target_label),
                                "source_label": int(parent["label"]),
                                "parent_image_id": str(parent["image_id"]),
                                "parent_label": int(parent["label"]),
                                "parent_abs_path": str(parent["parent_abs_path"]),
                                "parent_source_filename": str(parent.get("source_filename", Path(str(parent["relative_path"])).name)),
                                "cam": parent.get("cam", pd.NA),
                                "state": parent.get("state", pd.NA),
                                "cam_state": parent.get("cam_state", pd.NA),
                                "day": parent.get("day", pd.NA),
                                "sec": int(parent.get("sec", -1)),
                                "capture_id": str(parent["capture_id"]),
                                "block_index": int(parent.get("block_index", -1)),
                                "group_id": str(parent["group_id"]),
                                "synthetic_split": split_name,
                                "asset_split": split_name,
                                "release_path": str(release_path),
                                "task_seed": int(cfg.seed * 1_000_000 + global_uid),
                            }
                        )

                    attempts_by_target[target_label] += len(tasks)
                    for task, result in zip(tasks, executor.map(synthetic_task, tasks)):
                        if not result.get("accepted", False):
                            continue
                        if accepted_by_target[target_label] >= required:
                            Path(result["release_path"]).unlink(missing_ok=True)
                            continue
                        parent_id = str(task["parent_image_id"])
                        if parent_accept_counts[parent_id] >= max_per_parent:
                            Path(result["release_path"]).unlink(missing_ok=True)
                            continue

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
                                "relative_path": safe_relpath(Path(result["release_path"]), output_root),
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
                                "generator_version": cfg.generator_version,
                            }
                        )
                        accepted_by_target[target_label] += 1
                        parent_accept_counts[parent_id] += 1
                        progress.update(1)
                        if accepted_by_target[target_label] >= required:
                            break
        finally:
            progress.close()

    synth_df = pd.DataFrame(accepted_rows)
    expected_counts = cfg.target_synthetic_by_label(total=target_total)
    if count_by_label(synth_df) != expected_counts:
        raise RuntimeError(f"Label quotas not met for split {split_name}. Expected={expected_counts}, actual={count_by_label(synth_df)}")

    split_capture_ids = set(split_real_df["capture_id"].astype(str).unique().tolist())
    synth_capture_ids = set(synth_df["capture_id"].astype(str).unique().tolist())
    if not synth_capture_ids <= split_capture_ids:
        raise RuntimeError(f"Synthetic split {split_name} leaked outside parent captures")

    allowed_assets = set(split_asset_manifest_df["asset_id"].astype(str).tolist())
    used_assets = set(synth_df["asset_id"].dropna().astype(str).tolist())
    if not used_assets <= allowed_assets:
        raise RuntimeError(f"Synthetic split {split_name} used assets outside its assigned pool")

    return synth_df.sort_values(["label", "parent_image_id", "image_id"]).reset_index(drop=True), global_uid


def generate_all_synthetic_splits(
    cfg: SyntheticConfig,
    dataset_root: Path,
    output_root: Path,
    official_real_splits: Dict[str, pd.DataFrame],
    asset_manifest_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    partition_totals = cfg.target_synthetic_by_partition()
    synthetic_splits: Dict[str, pd.DataFrame] = {}
    next_uid = 0
    for split_name in SPLIT_NAMES:
        asset_subset = asset_manifest_df.loc[asset_manifest_df["asset_split"] == split_name].copy().reset_index(drop=True)
        if asset_subset.empty:
            raise RuntimeError(f"Asset subset for {split_name} is empty")
        synth_df, next_uid = generate_synthetic_images_for_split(
            cfg=cfg,
            dataset_root=dataset_root,
            output_root=output_root,
            split_name=split_name,
            split_real_df=official_real_splits[f"{split_name}_real_only"],
            split_asset_manifest_df=asset_subset,
            target_total=partition_totals[split_name],
            image_uid_start=next_uid,
        )
        synthetic_splits[split_name] = synth_df

    all_synth = pd.concat(list(synthetic_splits.values()), ignore_index=True)
    if len(all_synth) != cfg.target_synthetic_total:
        raise RuntimeError(f"Expected {cfg.target_synthetic_total} synthetic images, got {len(all_synth)}")
    if len(all_synth["image_id"].astype(str).unique()) != len(all_synth):
        raise RuntimeError("Synthetic image_id collision detected")
    return synthetic_splits


def assemble_official_splits(official_real_splits: Dict[str, pd.DataFrame], synthetic_splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    train_real_only = official_real_splits["train_real_only"].sort_values("image_id").reset_index(drop=True)
    val_real_only = official_real_splits["val_real_only"].sort_values("image_id").reset_index(drop=True)
    test_real_only = official_real_splits["test_real_only"].sort_values("image_id").reset_index(drop=True)

    train_synth = synthetic_splits["train"].sort_values("image_id").reset_index(drop=True)
    val_synth = synthetic_splits["val"].sort_values("image_id").reset_index(drop=True)
    test_synth = synthetic_splits["test"].sort_values("image_id").reset_index(drop=True)

    return {
        "train_synthetic_only": train_synth,
        "val_synthetic_only": val_synth,
        "test_synthetic_only": test_synth,
        "train_real_only": train_real_only,
        "val_real_only": val_real_only,
        "test_real_only": test_real_only,
        "train_real_plus_synthetic": pd.concat([train_real_only, train_synth], ignore_index=True).sort_values(["origin", "label", "image_id"]).reset_index(drop=True),
        "val_real_plus_synthetic": pd.concat([val_real_only, val_synth], ignore_index=True).sort_values(["origin", "label", "image_id"]).reset_index(drop=True),
        "test_real_plus_synthetic": pd.concat([test_real_only, test_synth], ignore_index=True).sort_values(["origin", "label", "image_id"]).reset_index(drop=True),
    }


def build_split_summary_rows(split_name: str, df: pd.DataFrame) -> Dict[str, object]:
    label_counts = df["label"].value_counts().reindex(list(REQUIRED_LABELS), fill_value=0)
    return {
        "split": split_name,
        "n_images": int(len(df)),
        "n_real": int((df["origin"] == "real").sum()) if "origin" in df.columns else 0,
        "n_synthetic": int((df["origin"] == "synthetic").sum()) if "origin" in df.columns else len(df),
        "n_captures": int(df["capture_id"].nunique()) if "capture_id" in df.columns else 0,
        "n_groups": int(df["group_id"].nunique()) if "group_id" in df.columns else 0,
        **{f"label_{label}": int(label_counts.loc[label]) for label in REQUIRED_LABELS},
    }


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    sanitize_dataframe(df).to_csv(path, index=False)


def write_outputs(
    output_root: Path,
    official_splits: Dict[str, pd.DataFrame],
    synthetic_splits: Dict[str, pd.DataFrame],
) -> None:
    meta_root = output_root / "metadata"
    split_root = output_root / "splits" / "official"
    ensure_dir(meta_root)
    ensure_dir(split_root)

    synth_df = pd.concat(list(synthetic_splits.values()), ignore_index=True).sort_values("image_id").reset_index(drop=True)
    synth_df = synth_df[[col for col in PUBLIC_METADATA_COLUMNS if col in synth_df.columns]]
    write_dataframe(synth_df, meta_root / "metadata_synthetic.csv")

    summary_rows = [build_split_summary_rows(name, df) for name, df in official_splits.items()]
    write_dataframe(pd.DataFrame(summary_rows), meta_root / "split_summary.csv")

    for split_name, df in official_splits.items():
        write_dataframe(df, split_root / f"{split_name}.csv")


def parse_args() -> SyntheticConfig:
    parser = argparse.ArgumentParser(description="Reproduce the public CleanCam synthetic subset from the Zenodo release")
    parser.add_argument("--dataset-root", required=True, help="Path to the public CleanCam dataset root")
    parser.add_argument("--output-root", required=True, help="Directory where regenerated synthetic outputs will be written")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--target-synthetic-total", type=int, default=3600)
    args = parser.parse_args()
    return SyntheticConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        seed=args.seed,
        num_workers=args.num_workers,
        target_synthetic_total=args.target_synthetic_total,
    )


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    dataset_root = Path(cfg.dataset_root).resolve()
    output_root = Path(cfg.output_root).resolve()
    ensure_dir(output_root)

    official_real_splits = load_official_real_splits(dataset_root)
    asset_manifest_df = load_asset_manifest(dataset_root)
    synthetic_splits = generate_all_synthetic_splits(cfg, dataset_root, output_root, official_real_splits, asset_manifest_df)
    official_splits = assemble_official_splits(official_real_splits, synthetic_splits)
    write_outputs(output_root, official_splits, synthetic_splits)

    print(f"Done. Synthetic outputs written to: {output_root}")
    for split_name in SPLIT_NAMES:
        df = synthetic_splits[split_name]
        print(f"synthetic_{split_name}: n={len(df)} labels={count_by_label(df)}")


if __name__ == "__main__":
    main()

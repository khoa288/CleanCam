"""Dataset integrity audit functions."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from cleancam_pipeline.utils.image import compute_phash, compute_sha256, hamming_distance


def audit_integrity(
    train_real: pd.DataFrame,
    val_real: pd.DataFrame,
    test_real: pd.DataFrame,
    train_aug: pd.DataFrame,
    val_aug: pd.DataFrame,
    test_aug: pd.DataFrame,
    metadata_real: pd.DataFrame,
) -> pd.DataFrame:
    """
    Audit dataset integrity across splits.

    Args:
        train_real: Training split (real only)
        val_real: Validation split (real only)
        test_real: Test split (real only)
        train_aug: Training split (real + synthetic)
        val_aug: Validation split (real + synthetic)
        test_aug: Test split (real + synthetic)
        metadata_real: Real image metadata

    Returns:
        DataFrame with integrity check results
    """
    rows: List[Dict[str, object]] = []

    def add(name: str, passed: bool, details: str) -> None:
        rows.append({"check": name, "passed": bool(passed), "details": details})

    def overlap(a: Iterable[object], b: Iterable[object]) -> set:
        return set(map(str, a)).intersection(set(map(str, b)))

    # Real split disjointness
    for field in ["image_id", "group_id", "capture_id"]:
        add(
            f"no_{field}_overlap_train_val_real",
            len(overlap(train_real[field], val_real[field])) == 0,
            f"overlap={len(overlap(train_real[field], val_real[field]))}",
        )
        add(
            f"no_{field}_overlap_train_test_real",
            len(overlap(train_real[field], test_real[field])) == 0,
            f"overlap={len(overlap(train_real[field], test_real[field]))}",
        )
        add(
            f"no_{field}_overlap_val_test_real",
            len(overlap(val_real[field], test_real[field])) == 0,
            f"overlap={len(overlap(val_real[field], test_real[field]))}",
        )

    # Augmented split sanity
    add(
        "train_aug_contains_all_train_real",
        set(map(str, train_real["image_id"]))
        <= set(map(str, train_aug["image_id"])),
        f"missing={len(set(map(str, train_real['image_id'])) - set(map(str, train_aug['image_id'])))}",
    )
    add(
        "val_aug_contains_all_val_real",
        set(map(str, val_real["image_id"])) <= set(map(str, val_aug["image_id"])),
        f"missing={len(set(map(str, val_real['image_id'])) - set(map(str, val_aug['image_id'])))}",
    )
    add(
        "test_aug_contains_all_test_real",
        set(map(str, test_real["image_id"])) <= set(map(str, test_aug["image_id"])),
        f"missing={len(set(map(str, test_real['image_id'])) - set(map(str, test_aug['image_id'])))}",
    )

    # Parent leakage checks
    def parent_leak_count(eval_df: pd.DataFrame) -> int:
        eval_ids = set(map(str, eval_df[eval_df["origin"] == "real"]["image_id"]))
        leaked = (
            train_aug[train_aug["origin"] == "synthetic"]["parent_image_id"]
            .astype(str)
            .isin(eval_ids)
            .sum()
        )
        return int(leaked)

    add(
        "no_training_parent_leak_into_val_real",
        parent_leak_count(val_real) == 0,
        f"n_leaks={parent_leak_count(val_real)}",
    )
    add(
        "no_training_parent_leak_into_test_real",
        parent_leak_count(test_real) == 0,
        f"n_leaks={parent_leak_count(test_real)}",
    )

    # Asset split leakage
    for split_name, df in [("train", train_aug), ("val", val_aug), ("test", test_aug)]:
        synth = df[df["origin"] == "synthetic"]
        if not synth.empty and "asset_split" in synth.columns:
            mismatched = (~synth["asset_split"].astype(str).eq(split_name)).sum()
            add(
                f"asset_split_consistent_{split_name}",
                int(mismatched) == 0,
                f"n_mismatched={int(mismatched)}",
            )

    # Synthetic parent capture consistency
    real_by_id = metadata_real.set_index("image_id")
    for split_name, df in [("train", train_aug), ("val", val_aug), ("test", test_aug)]:
        synth = df[df["origin"] == "synthetic"]
        split_captures = set(map(str, df[df["origin"] == "real"]["capture_id"]))
        bad = 0
        for parent_id in synth["parent_image_id"].dropna().astype(str):
            if parent_id not in real_by_id.index:
                bad += 1
                continue
            cap = str(real_by_id.loc[parent_id, "capture_id"])
            if cap not in split_captures:
                bad += 1
        add(
            f"parent_capture_consistent_{split_name}",
            bad == 0,
            f"n_bad={bad}",
        )

    return pd.DataFrame(rows)


def audit_exact_duplicates(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Audit for exact duplicate images using SHA256 hashes.

    Args:
        metadata: DataFrame with absolute_path and image_id columns

    Returns:
        DataFrame with duplicate findings
    """
    hash_to_ids: Dict[str, List[str]] = defaultdict(list)
    rows = []

    for _, row in metadata.iterrows():
        try:
            sha = compute_sha256(Path(row["absolute_path"]))
            hash_to_ids[sha].append(str(row["image_id"]))
        except Exception as e:
            rows.append(
                {"status": "error", "image_id": row["image_id"], "details": str(e)}
            )

    dup = {sha: ids for sha, ids in hash_to_ids.items() if len(ids) > 1}
    if not dup:
        return pd.DataFrame(
            [
                {
                    "status": "ok",
                    "n_duplicate_sha_groups": 0,
                    "details": "No exact duplicate file contents found.",
                }
            ]
        )

    for sha, ids in dup.items():
        rows.append(
            {
                "status": "duplicate_sha",
                "sha256": sha,
                "n_images": len(ids),
                "image_ids": "|".join(ids),
            }
        )

    return pd.DataFrame(rows)


def audit_near_duplicates_between_splits(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    threshold: int = 4,
    cap: Optional[int] = 1000,
) -> pd.DataFrame:
    """
    Audit for near-duplicate images between splits using perceptual hashing.

    Args:
        df_a: First split DataFrame
        df_b: Second split DataFrame
        threshold: Hamming distance threshold for near-duplicates
        cap: Maximum number of images to check per split (None for all)

    Returns:
        DataFrame with near-duplicate findings
    """
    a = df_a.copy()
    b = df_b.copy()

    if cap is not None:
        a = a.head(cap)
        b = b.head(cap)

    a_hashes = []
    for _, row in a.iterrows():
        try:
            a_hashes.append((str(row["image_id"]), compute_phash(Path(row["absolute_path"]))))
        except Exception:
            continue

    b_hashes = []
    for _, row in b.iterrows():
        try:
            b_hashes.append((str(row["image_id"]), compute_phash(Path(row["absolute_path"]))))
        except Exception:
            continue

    findings = []
    for a_id, a_hash in a_hashes:
        for b_id, b_hash in b_hashes:
            dist = hamming_distance(a_hash, b_hash)
            if dist <= threshold:
                findings.append(
                    {"image_id_a": a_id, "image_id_b": b_id, "hamming_distance": dist}
                )

    return pd.DataFrame(findings)

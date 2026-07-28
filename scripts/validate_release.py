#!/usr/bin/env python3
"""Validate the structure and cross-file relationships of a CleanCam release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


SPLIT_BASES = ("train", "val", "test")
EXPECTED_RELEASE_COUNTS = {
    "total_count": 22_572,
    "real_count": 18_972,
    "synthetic_count": 3_600,
}
SPLIT_VARIANTS = tuple(
    f"{base}_{variant}"
    for base in SPLIT_BASES
    for variant in ("real_only", "real_plus_synthetic")
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_unique(rows: Iterable[dict[str, str]], field: str, context: str) -> None:
    values = [row[field] for row in rows]
    repeated = [value for value, count in Counter(values).items() if count > 1]
    if repeated:
        raise ValueError(f"{context} repeats {field}: {sorted(repeated)[:5]}")


def ids(rows: Iterable[dict[str, str]]) -> set[str]:
    return {row["image_id"] for row in rows}


def validate_tables(root: Path) -> dict[str, object]:
    metadata = read_csv(root / "metadata/metadata.csv")
    real = read_csv(root / "metadata/metadata_real.csv")
    synthetic = read_csv(root / "metadata/metadata_synthetic.csv")
    assets = read_csv(root / "metadata/dirt_assets_manifest.csv")
    splits = {
        name: read_csv(root / f"splits/official/{name}.csv")
        for name in SPLIT_VARIANTS
    }

    for name, rows in (
        ("metadata.csv", metadata),
        ("metadata_real.csv", real),
        ("metadata_synthetic.csv", synthetic),
    ):
        require_unique(rows, "image_id", name)
        require_unique(rows, "relative_path", name)

    metadata_ids = ids(metadata)
    real_ids = ids(real)
    synthetic_ids = ids(synthetic)
    if real_ids & synthetic_ids:
        raise ValueError("Real and synthetic image identifiers overlap")
    if metadata_ids != real_ids | synthetic_ids:
        raise ValueError("metadata.csv is not the exact union of its two subsets")
    if any(row["origin"] != "real" for row in real):
        raise ValueError("metadata_real.csv contains a non-real record")
    if any(row["origin"] != "synthetic" for row in synthetic):
        raise ValueError("metadata_synthetic.csv contains a non-synthetic record")

    resolutions = {
        (int(row["width"]), int(row["height"]))
        for row in metadata
    }
    if resolutions != {(3072, 2048)}:
        raise ValueError(f"Unexpected released image resolutions: {sorted(resolutions)}")

    asset_lookup = {row["asset_id"]: row for row in assets}
    if len(asset_lookup) != 11:
        raise ValueError(f"Expected 11 asset records, found {len(asset_lookup)}")
    if set(asset_lookup) != {f"A_{index:04d}" for index in range(1, 12)}:
        raise ValueError("Asset identifiers are not the expected A_0001 through A_0011")

    real_split_by_id: dict[str, str] = {}
    synthetic_union: set[str] = set()
    capture_sets: dict[str, set[str]] = {}
    for base in SPLIT_BASES:
        real_rows = splits[f"{base}_real_only"]
        combined_rows = splits[f"{base}_real_plus_synthetic"]
        require_unique(real_rows, "image_id", f"{base}_real_only.csv")
        require_unique(combined_rows, "image_id", f"{base}_real_plus_synthetic.csv")
        base_real_ids = ids(real_rows)
        combined_real_ids = ids(
            row for row in combined_rows if row["origin"] == "real"
        )
        if base_real_ids != combined_real_ids:
            raise ValueError(f"Real rows differ between the two {base} split files")
        for image_id in base_real_ids:
            if image_id in real_split_by_id:
                raise ValueError(f"Real image is assigned to multiple splits: {image_id}")
            real_split_by_id[image_id] = base

        base_synthetic = [
            row for row in combined_rows if row["origin"] == "synthetic"
        ]
        base_synthetic_ids = ids(base_synthetic)
        if synthetic_union & base_synthetic_ids:
            raise ValueError(f"Synthetic rows overlap another split in {base}")
        synthetic_union |= base_synthetic_ids

        for row in base_synthetic:
            if row["parent_image_id"] not in base_real_ids:
                raise ValueError(
                    f"{row['image_id']} does not inherit its real parent's split"
                )
            if row["asset_id"] not in asset_lookup:
                raise ValueError(f"Unknown asset for {row['image_id']}: {row['asset_id']}")
            if asset_lookup[row["asset_id"]]["asset_split"] != base:
                raise ValueError(
                    f"{row['image_id']} uses an asset outside its {base} pool"
                )
        capture_sets[base] = {row["capture_id"] for row in real_rows}

    if set(real_split_by_id) != real_ids:
        raise ValueError("The real-only splits do not partition the real metadata")
    if synthetic_union != synthetic_ids:
        raise ValueError("The augmented splits do not partition the synthetic metadata")
    for left_index, left in enumerate(SPLIT_BASES):
        for right in SPLIT_BASES[left_index + 1 :]:
            overlap = capture_sets[left] & capture_sets[right]
            if overlap:
                raise ValueError(
                    f"Capture leakage between {left} and {right}: {sorted(overlap)[:5]}"
                )

    summary_path = root / "metadata/build_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "total_count": len(metadata),
        "real_count": len(real),
        "synthetic_count": len(synthetic),
    }
    for field, value in expected.items():
        if int(summary.get(field, -1)) != value:
            raise ValueError(
                f"build_summary.json {field}={summary.get(field)}; expected {value}"
            )
    if expected != EXPECTED_RELEASE_COUNTS:
        raise ValueError(
            f"Unexpected release inventory: {expected}; "
            f"expected {EXPECTED_RELEASE_COUNTS}"
        )

    missing_images = [
        row["relative_path"]
        for row in metadata
        if not (root / row["relative_path"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Metadata references missing images: {missing_images[:5]}")

    return {
        "release_tag": summary.get("release_tag"),
        "images": len(metadata),
        "real": len(real),
        "synthetic": len(synthetic),
        "assets": len(assets),
        "real_label_counts": dict(sorted(Counter(row["label"] for row in real).items())),
        "synthetic_label_counts": dict(
            sorted(Counter(row["label"] for row in synthetic).items())
        ),
        "captures": {
            base: len(capture_sets[base])
            for base in SPLIT_BASES
        },
    }


def validate_image_files(root: Path) -> dict[str, int]:
    from PIL import Image

    metadata = read_csv(root / "metadata/metadata.csv")
    for index, row in enumerate(metadata, start=1):
        path = root / row["relative_path"]
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB image, found {image.mode}: {path}")
            if image.size != (3072, 2048):
                raise ValueError(f"Unexpected image dimensions {image.size}: {path}")
        if index % 1000 == 0:
            print(f"Checked image headers: {index}/{len(metadata)}", flush=True)

    assets = read_csv(root / "metadata/dirt_assets_manifest.csv")
    for row in assets:
        path = root / row["release_relative_path"]
        with Image.open(path) as image:
            if image.mode != "RGBA":
                raise ValueError(f"Expected RGBA asset, found {image.mode}: {path}")
    return {"image_headers_checked": len(metadata), "asset_headers_checked": len(assets)}


def validate_manifest(root: Path) -> dict[str, int]:
    manifest_path = root / "metadata/file_manifest_sha256.csv"
    rows = read_csv(manifest_path)
    require_unique(rows, "relative_path", manifest_path.name)
    listed = {row["relative_path"] for row in rows}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if listed != actual:
        missing = sorted(listed - actual)[:5]
        unlisted = sorted(actual - listed)[:5]
        raise ValueError(
            f"Manifest inventory mismatch; missing={missing}, unlisted={unlisted}"
        )
    for index, row in enumerate(rows, start=1):
        path = root / row["relative_path"]
        if int(row["size_bytes"]) != path.stat().st_size:
            raise ValueError(f"Manifest size mismatch: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Manifest checksum mismatch: {path}")
        if index % 1000 == 0:
            print(f"Verified SHA-256: {index}/{len(rows)}", flush=True)
    return {"manifest_files_verified": len(rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path)
    parser.add_argument("--verify-image-files", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.release_root.resolve()
    try:
        report = validate_tables(root)
        if args.verify_image_files:
            report.update(validate_image_files(root))
        if args.verify_sha256:
            report.update(validate_manifest(root))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    report["status"] = "passed"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Finalize the validated CleanCam working tree as a public v2 dataset archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


FIXED_ZIP_TIME = (2026, 7, 27, 0, 0, 0)
EXPECTED_RELEASE_COUNTS = {
    "real_count": 18_972,
    "synthetic_count": 3_600,
    "total_count": 22_572,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["/bin/cp", "-cR", str(source), str(destination)], check=True)


def remove_macos_metadata(destination: Path) -> None:
    for path in destination.rglob(".DS_Store"):
        path.unlink()
    for path in destination.rglob("__MACOSX"):
        if path.is_dir():
            shutil.rmtree(path)


def enrich_asset_manifest(destination: Path, repo_root: Path) -> None:
    manifest_path = destination / "metadata/dirt_assets_manifest.csv"
    rows = read_csv(manifest_path)
    descriptions = {
        row["asset_id"]: row
        for row in read_csv(repo_root / "docs/deposit_assets.csv")
    }
    output: list[dict[str, object]] = []
    for row in rows:
        asset_id = row["asset_id"]
        if asset_id not in descriptions:
            raise ValueError(f"No documented description for {asset_id}")
        info = descriptions[asset_id]
        if row["asset_split"] != info["asset_split"]:
            raise ValueError(f"Asset split mismatch for {asset_id}")
        asset_path = destination / row["release_relative_path"]
        with Image.open(asset_path) as image:
            width, height = image.size
            mode = image.mode
        if mode != "RGBA":
            raise ValueError(f"Expected an RGBA deposit asset: {asset_path}")
        output.append(
            {
                "asset_id": asset_id,
                "asset_filename": row["asset_filename"],
                "release_relative_path": row["release_relative_path"],
                "asset_split": row["asset_split"],
                "description": info["description"],
                "preparation_category": info["preparation_category"],
                "width": width,
                "height": height,
                "mode": mode,
                "sha256": sha256_file(asset_path),
            }
        )
    write_csv(
        manifest_path,
        output,
        [
            "asset_id",
            "asset_filename",
            "release_relative_path",
            "asset_split",
            "description",
            "preparation_category",
            "width",
            "height",
            "mode",
            "sha256",
        ],
    )


def copy_public_materials(
    destination: Path,
    repo_root: Path,
    shader_screenshot: Path,
) -> None:
    code_root = destination / "code"
    code_root.mkdir(exist_ok=True)
    for name in (
        "build_cleancam_release.py",
        "validate_release.py",
    ):
        shutil.copy2(repo_root / "scripts" / name, code_root / name)

    documentation = destination / "documentation"
    documentation.mkdir(exist_ok=True)
    for name in (
        "acquisition_conditions.md",
        "data_dictionary.csv",
        "deposit_assets.md",
        "deposit_assets.csv",
        "label_taxonomy.csv",
        "quickstart.md",
    ):
        shutil.copy2(repo_root / "docs" / name, documentation / name)
    shutil.copy2(shader_screenshot, documentation / "blender_shader_network.jpeg")

    shutil.copy2(repo_root / "CHANGELOG.md", destination / "CHANGELOG.md")
    shutil.copy2(repo_root / "CITATION.cff", destination / "CITATION.cff")
    (destination / "LICENSE.txt").write_text(
        "CleanCam v2 dataset files are licensed under the Creative Commons "
        "Attribution 4.0 International license (CC BY 4.0).\n"
        "License terms: https://creativecommons.org/licenses/by/4.0/\n",
        encoding="utf-8",
    )


def update_build_summary(destination: Path, release_tag: str) -> None:
    path = destination / "metadata/build_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    for field, expected in EXPECTED_RELEASE_COUNTS.items():
        actual = int(summary.get(field, -1))
        if actual != expected:
            raise ValueError(
                f"Expected {field}={expected:,}, found {actual:,} in source release"
            )
    summary["release_tag"] = release_tag
    summary["archive_filename"] = "CleanCam_v2.zip"
    summary["image_resolution"] = {"width": 3072, "height": 2048}
    summary["deposit_asset_count"] = 11
    summary["documentation"] = [
        "documentation/acquisition_conditions.md",
        "documentation/data_dictionary.csv",
        "documentation/deposit_assets.md",
        "documentation/deposit_assets.csv",
        "documentation/label_taxonomy.csv",
        "documentation/quickstart.md",
    ]
    summary["validation"] = {
        "metadata_image_inventory": "passed",
        "capture_disjoint_splits": "passed",
        "synthetic_parent_split_inheritance": "passed",
        "disjoint_asset_pools": "passed",
        "consistent_rgb_resolution": "3072 x 2048",
    }
    summary["finalized_utc"] = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_release_contents(destination: Path) -> None:
    rows = [
        {
            "path": path.relative_to(destination).as_posix(),
            "type": "directory" if path.is_dir() else "file",
            "description": "",
        }
        for path in sorted(destination.rglob("*"))
        if path.relative_to(destination).parts[0] not in {"images"}
    ]
    write_csv(
        destination / "documentation/release_contents.csv",
        rows,
        ["path", "type", "description"],
    )


def write_readme(destination: Path) -> None:
    text = """# CleanCam v2

CleanCam is a labelled underwater-image dataset for assessing material attached
to camera viewports in aquaculture monitoring. The release contains 18,972
field images and 3,600 split-consistent synthetic images, for 22,572 RGB JPEG
images at 3072 × 2048 pixels.

## Start here

- `documentation/quickstart.md`: installation and first-use example.
- `documentation/data_dictionary.csv`: field-level definitions.
- `documentation/label_taxonomy.csv`: five-level ordinal label definitions.
- `documentation/acquisition_conditions.md`: camera and collection conditions.
- `documentation/deposit_assets.md`: asset preparation and overlay method.
- `metadata/metadata.csv`: master image inventory.
- `splits/official/`: capture-disjoint split files.
- `metadata/file_manifest_sha256.csv`: complete file checksums.

Real and synthetic images are stored separately under `images/`. Synthetic
images inherit the split of their real parent, and the 11 RGBA deposit assets
are assigned to disjoint train, validation, and test pools.

Dataset v2.0.0 DOI: https://doi.org/10.5281/zenodo.21515620

Concept DOI (all versions): https://doi.org/10.5281/zenodo.18952473

License: CC BY 4.0. See `LICENSE.txt`.
"""
    (destination / "README.md").write_text(text, encoding="utf-8")


def write_file_manifest(destination: Path) -> None:
    manifest_path = destination / "metadata/file_manifest_sha256.csv"
    rows = []
    paths = [
        path
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    for index, path in enumerate(paths, start=1):
        rows.append(
            {
                "relative_path": path.relative_to(destination).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if index % 1000 == 0:
            print(f"Hashed release files: {index}/{len(paths)}", flush=True)
    write_csv(
        manifest_path,
        rows,
        ["relative_path", "size_bytes", "sha256"],
    )


def build_zip(source: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Archive already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [path for path in sorted(source.rglob("*")) if path.is_file()]
    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for index, path in enumerate(paths, start=1):
            relative = Path(source.name) / path.relative_to(source)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as handle:
                archive.writestr(info, handle.read())
            if index % 1000 == 0:
                print(f"Archived files: {index}/{len(paths)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--output-release", type=Path, required=True)
    parser.add_argument("--zip-output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--shader-screenshot", type=Path, required=True)
    parser.add_argument("--release-tag", default="v2.0.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_release.resolve()
    destination = args.output_release.resolve()
    repo_root = args.repo_root.resolve()
    screenshot = args.shader_screenshot.resolve()
    try:
        if not source.is_dir():
            raise FileNotFoundError(source)
        if not screenshot.is_file():
            raise FileNotFoundError(screenshot)
        copy_tree(source, destination)
        remove_macos_metadata(destination)
        copy_public_materials(destination, repo_root, screenshot)
        enrich_asset_manifest(destination, repo_root)
        update_build_summary(destination, args.release_tag)
        write_readme(destination)
        write_release_contents(destination)
        write_file_manifest(destination)
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/validate_release.py"),
                str(destination),
                "--verify-image-files",
            ],
            check=True,
        )
        build_zip(destination, args.zip_output.resolve())
        archive_sha256 = sha256_file(args.zip_output.resolve())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "release": str(destination),
                "archive": str(args.zip_output.resolve()),
                "archive_bytes": args.zip_output.resolve().stat().st_size,
                "archive_sha256": archive_sha256,
                "status": "passed",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

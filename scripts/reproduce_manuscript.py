#!/usr/bin/env python3
"""Reproduce CleanCam manuscript Figures 2–6 and Tables 1–3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import PCA
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cleancam_pipeline.analysis.synthetic import (
    brightness_mean,
    grayscale_entropy,
    laplacian_variance,
    rms_contrast,
    tenengrad,
)


LABELS = (1, 2, 3, 4, 5)
BLUE = "#2878B5"
ORANGE = "#E66101"
GREEN = "#2A9D8F"
GOLD = "#E9C46A"
INK = "#263238"
LIGHT_BLUE = "#E8F2F8"
LIGHT_GREEN = "#E8F5F2"
WORKFLOW_BOX_PAD = 0.012


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_release(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(root / "metadata/metadata.csv", low_memory=False)
    real = pd.read_csv(root / "metadata/metadata_real.csv", low_memory=False)
    synthetic = pd.read_csv(
        root / "metadata/metadata_synthetic.csv", low_memory=False
    )
    if len(metadata) != len(real) + len(synthetic):
        raise ValueError("Master metadata is not the union of real and synthetic rows")
    if set(zip(metadata["width"], metadata["height"])) != {(3072, 2048)}:
        raise ValueError("Manuscript reproduction expects 3072 x 2048 images")
    for frame in (metadata, real, synthetic):
        frame["absolute_path"] = frame["relative_path"].map(
            lambda value: str(root / str(value))
        )
    return metadata, real, synthetic


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def selected_rows(
    metadata: pd.DataFrame,
    image_ids: list[str],
    expected_origin: str,
    expected_label: int,
) -> pd.DataFrame:
    lookup = metadata.set_index("image_id", drop=False)
    missing = [image_id for image_id in image_ids if image_id not in lookup.index]
    if missing:
        raise KeyError(f"Configured manuscript examples are absent: {missing}")
    rows = lookup.loc[image_ids].copy()
    if set(rows["origin"]) != {expected_origin}:
        raise ValueError(f"Configured examples do not all have origin={expected_origin}")
    if set(rows["label"].astype(int)) != {expected_label}:
        raise ValueError(f"Configured examples do not all have label={expected_label}")
    return rows


def figure_2_real_examples(
    metadata: pd.DataFrame,
    config: dict[str, object],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(5, 3, figsize=(7.687, 10.0))
    for row_index, label in enumerate(LABELS):
        image_ids = config["figure_2_real"][str(label)]
        rows = selected_rows(metadata, image_ids, "real", label)
        for column_index, (_, row) in enumerate(rows.iterrows()):
            axis = axes[row_index, column_index]
            axis.imshow(read_rgb(Path(row["absolute_path"])))
            axis.set_title(f"L{label} | real", pad=4)
            axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, wspace=0.03, hspace=0.12)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def figure_3_composition(real: pd.DataFrame, output_path: Path) -> None:
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(10.5, 7.04), gridspec_kw={"width_ratios": [0.72, 1.28]}
    )

    counts = real["label"].astype(int).value_counts().reindex(LABELS, fill_value=0)
    total = int(counts.sum())
    colors = ["#7FB3D5", "#76C7C0", "#F4D35E", "#EE964B", "#C44536"]
    label_names = ["clean", "light", "moderate", "severe", "heavy"]
    positions = np.arange(len(LABELS))
    bars = left.barh(positions, counts.values, color=colors, height=0.72)
    left.set_yticks(positions)
    left.set_yticklabels(
        [f"L{label}\n{name}" for label, name in zip(LABELS, label_names)]
    )
    left.invert_yaxis()
    left.set_xlabel("Number of real images")
    left.set_title("(a) Real-image label distribution", pad=10)
    left.grid(axis="x", alpha=0.22)
    left.set_axisbelow(True)
    for bar, value in zip(bars, counts.values):
        left.text(
            value + max(counts.values) * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}\n({value / total:.1%})",
            ha="left",
            va="center",
            fontsize=8,
        )
    left.set_xlim(0, max(counts.values) * 1.22)

    heatmap = (
        real.groupby(["cam_state", "day"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )
    preferred_rows = [
        "cambot1__morning",
        "cambot1__evening",
        "cambot2__morning",
        "cambot2__evening",
    ]
    heatmap = heatmap.reindex(
        [name for name in preferred_rows if name in heatmap.index]
        + [name for name in heatmap.index if name not in preferred_rows]
    )
    heatmap = heatmap.T
    image = right.imshow(heatmap.to_numpy(), aspect="auto", cmap="Blues")
    right.set_title("(b) Sampled images by date and camera/session", pad=10)
    right.set_yticks(np.arange(len(heatmap.index)))
    right.set_yticklabels(
        [
            pd.to_datetime(str(value), format="%Y%m%d").strftime("%b %d")
            for value in heatmap.index
        ]
    )
    session_labels = [
        value.replace("cambot", "CAMBOT").replace("__morning", " AM").replace(
            "__evening", " PM"
        )
        for value in heatmap.columns
    ]
    right.set_xticks(np.arange(len(session_labels)))
    right.set_xticklabels(session_labels, rotation=32, ha="right")
    right.set_xlabel("Camera / session")
    right.set_xticks(np.arange(-0.5, len(session_labels), 1), minor=True)
    right.set_yticks(np.arange(-0.5, len(heatmap.index), 1), minor=True)
    right.grid(which="minor", color="white", linewidth=0.7)
    right.tick_params(which="minor", bottom=False, left=False)
    maximum = float(heatmap.to_numpy().max())
    for row_index in range(heatmap.shape[0]):
        for column_index in range(heatmap.shape[1]):
            value = int(heatmap.iloc[row_index, column_index])
            if value:
                right.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if value > maximum * 0.48 else INK,
                )
    colorbar = figure.colorbar(image, ax=right, fraction=0.035, pad=0.02)
    colorbar.set_label("Images")

    figure.tight_layout(w_pad=2.4)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def figure_4_synthetic_examples(
    metadata: pd.DataFrame,
    config: dict[str, object],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(10.0, 7.675))
    for row_index, label in enumerate((3, 4, 5)):
        image_ids = config["figure_4_synthetic"][str(label)]
        rows = selected_rows(metadata, image_ids, "synthetic", label)
        for column_index, (_, row) in enumerate(rows.iterrows()):
            axis = axes[row_index, column_index]
            axis.imshow(read_rgb(Path(row["absolute_path"])))
            axis.set_title(f"L{label} | synthetic", pad=5)
            axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.03, hspace=0.14)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _initialize_worker() -> None:
    cv2.setNumThreads(1)


def _extract_stats(task: tuple[str, str, int, str]) -> dict[str, object]:
    image_id, group, label, path = task
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {path}")
    return {
        "image_id": image_id,
        "group": group,
        "label": label,
        "laplacian_var": laplacian_variance(gray),
        "tenengrad": tenengrad(gray),
        "rms_contrast": rms_contrast(gray),
        "entropy": grayscale_entropy(gray),
        "brightness_mean": brightness_mean(gray),
    }


def expected_stats_tasks(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> list[tuple[str, str, int, str]]:
    frames = (
        (real[real["label"].isin([4, 5])], "real_severe"),
        (synthetic[synthetic["label"].isin([4, 5])], "synthetic_severe"),
    )
    tasks = []
    for frame, group in frames:
        for _, row in frame.sort_values("image_id").iterrows():
            tasks.append(
                (
                    str(row["image_id"]),
                    group,
                    int(row["label"]),
                    str(row["absolute_path"]),
                )
            )
    return tasks


def load_or_compute_stats(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    output_path: Path,
    workers: int,
    reuse: bool,
) -> pd.DataFrame:
    tasks = expected_stats_tasks(real, synthetic)
    expected_ids = {task[0] for task in tasks}
    if reuse and output_path.is_file():
        cached = pd.read_csv(output_path)
        if set(cached["image_id"]) == expected_ids:
            print(f"Reusing {len(cached):,} cached image-statistic rows", flush=True)
            return cached

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
    ) as executor:
        rows = list(
            tqdm(
                executor.map(_extract_stats, tasks, chunksize=8),
                total=len(tasks),
                desc="Low-level statistics",
            )
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(output_path, index=False, lineterminator="\n")
    return stats


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(list(values), dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def figure_5_characterization(
    stats: pd.DataFrame,
    figure_path: Path,
    table_root: Path,
) -> None:
    metrics = [
        "laplacian_var",
        "tenengrad",
        "rms_contrast",
        "entropy",
        "brightness_mean",
    ]
    x = stats[metrics].to_numpy(dtype=np.float64)
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0, ddof=0)
    standardized = (x - feature_mean) / np.maximum(feature_std, 1e-12)
    pca = PCA(n_components=2, svd_solver="full")
    scores = pca.fit_transform(standardized)

    pd.DataFrame(
        {
            "feature": metrics,
            "mean": feature_mean,
            "population_std": feature_std,
        }
    ).to_csv(table_root / "figure_5_feature_standardization.csv", index=False)
    pd.DataFrame(
        pca.components_.T,
        index=metrics,
        columns=["PC1", "PC2"],
    ).rename_axis("feature").reset_index().to_csv(
        table_root / "figure_5_pca_loadings.csv", index=False
    )
    pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(table_root / "figure_5_pca_explained_variance.csv", index=False)

    figure = plt.figure(figsize=(12.2, 7.0))
    grid = figure.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], wspace=0.32, hspace=0.38)
    pca_axis = figure.add_subplot(grid[:, 0])
    ecdf_axes = [
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[1, 2]),
    ]
    groups = (
        ("real_severe", "Real severe", BLUE),
        ("synthetic_severe", "Synthetic severe", ORANGE),
    )
    for group, display, color in groups:
        mask = stats["group"].eq(group).to_numpy()
        pca_axis.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=9,
            alpha=0.52,
            linewidths=0,
            color=color,
            label=display,
            rasterized=True,
        )
    pca_axis.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
    )
    pca_axis.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    )
    pca_axis.set_title("(a) PCA of standardized low-level statistics", pad=10)
    pca_axis.grid(alpha=0.22)
    pca_axis.legend(loc="upper left")

    display_metrics = [
        ("laplacian_var", "Laplacian variance"),
        ("tenengrad", "Tenengrad"),
        ("rms_contrast", "RMS contrast"),
        ("entropy", "Entropy (bits)"),
    ]
    for axis, (metric, title) in zip(ecdf_axes, display_metrics):
        for group, display, color in groups:
            values = stats.loc[stats["group"] == group, metric].dropna()
            x_values, y_values = ecdf(values)
            axis.plot(x_values, y_values, color=color, linewidth=1.8, label=display)
        axis.set_title(title)
        axis.set_xlabel("Value")
        axis.set_ylabel("ECDF")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.22)
    ecdf_axes[1].legend(loc="lower right")
    figure.text(
        0.70,
        0.035,
        "(b) Empirical cumulative distributions of selected features",
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.13)
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)


def add_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad={WORKFLOW_BOX_PAD},rounding_size=0.018",
        linewidth=1.25,
        edgecolor=color,
        facecolor="white",
        zorder=2,
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + height * 0.65,
        title,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=INK,
        zorder=3,
    )
    axis.text(
        x + width / 2,
        y + height * 0.30,
        detail,
        ha="center",
        va="center",
        fontsize=7.4,
        color="#455A64",
        linespacing=1.15,
        zorder=3,
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#607D8B",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=color,
            connectionstyle="arc3,rad=0",
            shrinkA=0,
            shrinkB=0,
            capstyle="butt",
            zorder=1,
        )
    )


def figure_6_workflow(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.6, 5.1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(
        FancyBboxPatch(
            (0.015, 0.55),
            0.97,
            0.39,
            boxstyle="round,pad=0.01",
            facecolor=LIGHT_BLUE,
            edgecolor="none",
            zorder=0,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (0.255, 0.06),
            0.73,
            0.36,
            boxstyle="round,pad=0.01",
            facecolor=LIGHT_GREEN,
            edgecolor="none",
            zorder=0,
        )
    )
    axis.text(0.03, 0.905, "Field-image path", fontweight="bold", color=BLUE)
    axis.text(0.27, 0.385, "Synthetic augmentation and release path", fontweight="bold", color=GREEN)

    width, height = 0.175, 0.20
    top_x = [0.035, 0.275, 0.515, 0.755]
    top_boxes = [
        ("Underwater videos", "Two fixed cameras\n3072 × 2048"),
        ("Frame sampling", "Every 10 s\n≈53 h valid footage"),
        ("Temporal annotation", "Three independent annotators\nfive ordered levels"),
        ("Consensus real set", "18,972 images\ncapture identifiers"),
    ]
    for x, (title, detail) in zip(top_x, top_boxes):
        add_box(axis, (x, 0.635), width, height, title, detail, BLUE)
    for index in range(3):
        add_arrow(
            axis,
            (top_x[index] + width + WORKFLOW_BOX_PAD, 0.735),
            (top_x[index + 1] - WORKFLOW_BOX_PAD, 0.735),
        )

    bottom_x = [0.275, 0.515, 0.755]
    bottom_boxes = [
        ("Capture-disjoint splits", "Train / validation / test\nreal-only lists"),
        ("Viewport overlays", "11 disjoint RGBA assets\nrecorded parameters"),
        ("Validated v2 release", "22,572 images\nmetadata + checksums"),
    ]
    for x, (title, detail) in zip(bottom_x, bottom_boxes):
        add_box(axis, (x, 0.13), width, height, title, detail, GREEN)
    for index in range(2):
        add_arrow(
            axis,
            (bottom_x[index] + width + WORKFLOW_BOX_PAD, 0.23),
            (bottom_x[index + 1] - WORKFLOW_BOX_PAD, 0.23),
        )
    axis.plot(
        [0.8425, 0.8425, 0.24, 0.24],
        [0.635 - WORKFLOW_BOX_PAD, 0.49, 0.49, 0.23],
        color=GREEN,
        linewidth=1.25,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=1,
    )
    add_arrow(
        axis,
        (0.24, 0.23),
        (bottom_x[0] - WORKFLOW_BOX_PAD, 0.23),
        color=GREEN,
    )
    axis.text(
        0.55,
        0.505,
        "split assignment inherited by every synthetic child",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#39766B",
    )
    axis.text(
        0.602,
        0.085,
        "Official lists and validation prevent capture, parent, and asset-pool leakage.",
        ha="center",
        fontsize=8,
        color="#455A64",
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_tables(
    release_root: Path,
    metadata: pd.DataFrame,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    table_root: Path,
) -> None:
    release_rows = [
        ("images/real/", "Field-captured RGB JPEG images organized by level."),
        ("images/synthetic/", "Generated RGB JPEG images for Levels 3–5."),
        ("assets/dirt_assets/", "Eleven RGBA deposit assets."),
        ("metadata/", "Image inventories, asset manifest, summaries, and checksums."),
        ("splits/official/", "Capture-disjoint real-only and augmented split files."),
        ("documentation/", "Quickstart, data dictionary, taxonomy, and method details."),
        ("code/", "Release builder and validation scripts."),
    ]
    pd.DataFrame(release_rows, columns=["path", "contents_and_purpose"]).to_csv(
        table_root / "Table_1_Release_contents.csv", index=False
    )

    rows = []
    for subset, frame in (
        ("Real", real),
        ("Synthetic", synthetic),
        ("Combined", metadata),
    ):
        counts = frame["label"].astype(int).value_counts().reindex(LABELS, fill_value=0)
        rows.append(
            {
                "Subset": subset,
                "Total": len(frame),
                **{f"L{label}": int(counts[label]) for label in LABELS},
            }
        )
    pd.DataFrame(rows).to_csv(table_root / "Table_2_Dataset_composition.csv", index=False)
    taxonomy_path = release_root / "documentation/label_taxonomy.csv"
    if not taxonomy_path.is_file():
        taxonomy_path = Path(__file__).resolve().parents[1] / "docs/label_taxonomy.csv"
    pd.read_csv(taxonomy_path).to_csv(table_root / "Table_3_Label_taxonomy.csv", index=False)


def write_manifest(
    output_root: Path,
    release_root: Path,
    config_path: Path,
    config: dict[str, object],
    workers: int,
) -> None:
    manifest_path = output_root / "reproducibility_manifest.json"
    outputs = [
        {
            "relative_path": path.relative_to(output_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "opencv-python-headless",
        "Pillow",
        "matplotlib",
        "scikit-learn",
        "scipy",
    ):
        versions[package] = importlib.metadata.version(package)
    payload = {
        "release_root": str(release_root.resolve()),
        "release_build_summary_sha256": sha256_file(
            release_root / "metadata/build_summary.json"
        ),
        "example_config": str(config_path),
        "example_config_sha256": sha256_file(config_path),
        "selected_image_ids": config,
        "statistics_scope": "all released real and synthetic images labelled L4 or L5",
        "statistics_workers": workers,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--examples-config",
        type=Path,
        default=repo_root / "configs/manuscript_examples.json",
    )
    parser.add_argument(
        "--stats-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--no-reuse-stats",
        action="store_true",
        help="Recompute the image-level low-level statistics even when a complete cache exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stats_workers < 1:
        raise ValueError("--stats-workers must be positive")
    release_root = args.release_root.resolve()
    output_root = args.output_root.resolve()
    figure_root = output_root / "figures"
    table_root = output_root / "tables"
    figure_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    config = json.loads(args.examples_config.read_text(encoding="utf-8"))
    metadata, real, synthetic = load_release(release_root)
    figure_2_real_examples(
        metadata,
        config,
        figure_root / "Figure_2_Representative_real_images.png",
    )
    figure_3_composition(
        real,
        figure_root / "Figure_3_Dataset_composition.png",
    )
    figure_4_synthetic_examples(
        metadata,
        config,
        figure_root / "Figure_4_Synthetic_examples.png",
    )
    stats_path = table_root / "figure_5_low_level_statistics.csv"
    stats = load_or_compute_stats(
        real,
        synthetic,
        stats_path,
        args.stats_workers,
        reuse=not args.no_reuse_stats,
    )
    figure_5_characterization(
        stats,
        figure_root / "Figure_5_Synthetic_characterization.png",
        table_root,
    )
    figure_6_workflow(
        figure_root / "Figure_6_Reproducible_workflow.png",
    )
    write_tables(release_root, metadata, real, synthetic, table_root)
    write_manifest(
        output_root,
        release_root,
        args.examples_config,
        config,
        args.stats_workers,
    )
    print(f"Manuscript outputs written to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plotting functions for CleanCam analysis."""

import random
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from cleancam_pipeline.core.constants import LABELS
from cleancam_pipeline.utils.image import load_image_rgb


def plot_label_distribution_overall(
    metadata: pd.DataFrame,
    metadata_real: pd.DataFrame,
    metadata_synth: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Plot overall label distribution across subsets.

    Args:
        metadata: Combined metadata
        metadata_real: Real subset metadata
        metadata_synth: Synthetic subset metadata
        out_path: Output file path
    """
    plt.figure(figsize=(10, 5))
    subsets = [
        ("Real", metadata_real),
        ("Synthetic", metadata_synth),
        ("Combined", metadata),
    ]
    x = np.arange(len(LABELS))
    width = 0.25
    for i, (name, df) in enumerate(subsets):
        counts = df["label"].value_counts().reindex(LABELS, fill_value=0).to_numpy()
        plt.bar(x + (i - 1) * width, counts, width=width, label=name)
    plt.xticks(x, [f"L{l}" for l in LABELS])
    plt.ylabel("Number of images")
    plt.title("CleanCam label distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_split_label_distribution(
    official_splits: Dict[str, pd.DataFrame], out_path: Path
) -> None:
    """
    Plot label distribution for each official split.

    Args:
        official_splits: Dictionary of split name to DataFrame
        out_path: Output file path
    """
    split_names = list(official_splits.keys())
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()
    for ax, split_name in zip(axes, split_names):
        df = official_splits[split_name]
        counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
        ax.bar([f"L{l}" for l in LABELS], counts.values)
        ax.set_title(split_name)
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_real_counts_by_day_camera_session(
    metadata_real: pd.DataFrame, out_path: Path
) -> None:
    """
    Plot real image counts by day and camera/session.

    Args:
        metadata_real: Real subset metadata
        out_path: Output file path
    """
    df = metadata_real.copy()
    pivot = df.groupby(["day", "cam_state"]).size().unstack(fill_value=0).sort_index()
    plt.figure(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, figsize=(14, 6))
    plt.ylabel("Number of real images")
    plt.title("Real-image counts by day and camera/session")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_example_grid(
    metadata: pd.DataFrame,
    out_path: Path,
    origin: str,
    max_per_label: int = 3,
    seed: int = 42,
) -> None:
    """
    Create grid of example images.

    Args:
        metadata: Metadata DataFrame
        out_path: Output file path
        origin: Origin filter ('real' or 'synthetic')
        max_per_label: Maximum examples per label
        seed: Random seed
    """
    rng = random.Random(seed)
    df = metadata[metadata["origin"] == origin].copy()
    chosen_rows = []
    for label in LABELS:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        idxs = list(sub.index)
        picked = rng.sample(idxs, min(max_per_label, len(idxs)))
        chosen_rows.append(sub.loc[picked])
    if not chosen_rows:
        return
    chosen = pd.concat(chosen_rows, ignore_index=True)
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
            ax = axes[label - 1, j]
            img = load_image_rgb(Path(sub.loc[j, "absolute_path"]))
            ax.imshow(img)
            ax.set_title(f"L{label} | {sub.loc[j, 'origin']}")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_synthetic_source_target(
    metadata_synth: pd.DataFrame, out_path: Path
) -> None:
    """
    Plot synthetic source-to-target label counts.

    Args:
        metadata_synth: Synthetic subset metadata
        out_path: Output file path
    """
    synth = metadata_synth.copy()
    if synth.empty:
        return
    table = synth.groupby(["source_label", "label"]).size().unstack(fill_value=0).sort_index()
    ax = table.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("Parent label")
    ax.set_ylabel("Number of synthetic images")
    ax.set_title("Synthetic source-to-target counts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_hist_comparison(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, out_path: Path
) -> None:
    """
    Plot histogram comparison of low-level statistics.

    Args:
        real_df: Real image statistics
        synth_df: Synthetic image statistics
        out_path: Output file path
    """
    metrics = ["laplacian_var", "tenengrad", "rms_contrast", "entropy"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        if metric in real_df.columns and not real_df.empty:
            ax.hist(
                real_df[metric].dropna(),
                bins=40,
                density=True,
                alpha=0.6,
                label="real severe",
            )
        if metric in synth_df.columns and not synth_df.empty:
            ax.hist(
                synth_df[metric].dropna(),
                bins=40,
                density=True,
                alpha=0.6,
                label="synthetic severe",
            )
        ax.set_title(metric)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pca_stats(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, out_path: Path
) -> None:
    """
    Plot PCA of low-level statistics.

    Args:
        real_df: Real image statistics
        synth_df: Synthetic image statistics
        out_path: Output file path
    """
    metrics = ["laplacian_var", "tenengrad", "rms_contrast", "entropy", "brightness_mean"]
    if real_df.empty or synth_df.empty:
        return
    combined = pd.concat(
        [
            real_df[metrics].assign(group="real severe"),
            synth_df[metrics].assign(group="synthetic severe"),
        ],
        ignore_index=True,
    ).dropna()
    if len(combined) < 4:
        return
    X = combined[metrics].to_numpy(dtype=np.float64)
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-8)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    combined["pc1"] = Z[:, 0]
    combined["pc2"] = Z[:, 1]
    plt.figure(figsize=(7, 6))
    for group_name, sub in combined.groupby("group"):
        plt.scatter(sub["pc1"], sub["pc2"], s=14, alpha=0.6, label=group_name)
    plt.xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    plt.ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    plt.title("PCA over low-level statistics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_setting_metric(
    main_df: pd.DataFrame, metric_col: str, out_path: Path, title: str
) -> None:
    """
    Plot metric comparison across models and settings.

    Args:
        main_df: Main results DataFrame
        metric_col: Column name of metric to plot
        out_path: Output file path
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    models_order = list(main_df["model"].unique())
    settings_order = list(main_df["setting"].unique())
    x = np.arange(len(models_order))
    width = 0.18
    for i, setting in enumerate(settings_order):
        sub = main_df[main_df["setting"] == setting].set_index("model")
        vals = [sub.loc[m, metric_col] if m in sub.index else np.nan for m in models_order]
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=setting)
    plt.xticks(x, models_order)
    plt.ylabel(metric_col)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

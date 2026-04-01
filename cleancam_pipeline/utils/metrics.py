"""Metric computation utilities."""

from typing import Dict

import numpy as np
import pandas as pd

from cleancam_pipeline.core.constants import LABELS


def count_by_label(df: pd.DataFrame) -> Dict[int, int]:
    """Count samples per label, ensuring all labels are present."""
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    return {int(label): int(counts.loc[label]) for label in LABELS}


def maybe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract numeric series from column, handling missing columns."""
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def compute_within_one_accuracy(
    true_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """Compute accuracy allowing predictions within ±1 of true label."""
    return float(np.mean(np.abs(true_labels - pred_labels) <= 1))


def format_label_distribution(df: pd.DataFrame) -> str:
    """Format label distribution as compact string."""
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    return " | ".join([f"L{label}:{int(counts.loc[label])}" for label in LABELS])

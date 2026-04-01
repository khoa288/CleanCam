"""Dataset characterization functions."""

from typing import Dict

import pandas as pd

from cleancam_pipeline.core.constants import LABELS
from cleancam_pipeline.utils.metrics import count_by_label


def make_release_composition_table(
    metadata: pd.DataFrame,
    metadata_real: pd.DataFrame,
    metadata_synth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create table showing composition of release by subset.

    Args:
        metadata: Combined metadata
        metadata_real: Real subset metadata
        metadata_synth: Synthetic subset metadata

    Returns:
        DataFrame with composition statistics
    """
    rows = []
    for subset_name, df in [
        ("real", metadata_real),
        ("synthetic", metadata_synth),
        ("combined", metadata),
    ]:
        row = {"subset": subset_name, "n_images": int(len(df))}
        for label, count in count_by_label(df).items():
            row[f"label_{label}"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def make_split_composition_table(
    official_splits: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create table showing composition of official splits.

    Args:
        official_splits: Dictionary of split name to DataFrame

    Returns:
        DataFrame with split composition statistics
    """
    rows = []
    for split_name, df in official_splits.items():
        row = {
            "split": split_name,
            "n_images": int(len(df)),
            "n_real": int((df["origin"] == "real").sum()),
            "n_synthetic": int((df["origin"] == "synthetic").sum()),
            "n_captures": int(df["capture_id"].nunique()),
            "n_groups": int(df["group_id"].nunique()),
        }
        for label, count in count_by_label(df).items():
            row[f"label_{label}"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def make_group_count_table(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Create table counting images by group field and label.

    Args:
        df: DataFrame with label and grouping field
        field: Name of grouping field (e.g., 'cam', 'day', 'capture_id')

    Returns:
        DataFrame with counts by group and label
    """
    out = (
        df.groupby([field, "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    out.columns = [field] + [f"label_{label}" for label in LABELS]
    out["n_images"] = out[[f"label_{label}" for label in LABELS]].sum(axis=1)
    return out

"""Annotation agreement analysis orchestrator."""

from pathlib import Path

import pandas as pd

from cleancam_pipeline.analysis.annotation import compute_annotation_agreement
from cleancam_pipeline.utils.io import OutputManager, save_json, save_table


def run_annotation(annotation_csv: Path, out: OutputManager) -> None:
    """
    Run annotation agreement analysis.

    Args:
        annotation_csv: Path to annotation CSV file
        out: Output manager
    """
    ann = pd.read_csv(annotation_csv)
    pair_df, conf_df, image_df, summary = compute_annotation_agreement(ann)

    save_table(
        pair_df,
        out.tables_root / "annotation_pairs.csv",
        out.tables_root / "annotation_pairs.tex",
    )
    if not conf_df.empty:
        save_table(
            conf_df.reset_index(),
            out.tables_root / "annotation_first_pair_confusion.csv",
        )
    save_table(
        image_df, out.tables_root / "annotation_image_level_disagreement.csv"
    )
    save_json(summary, out.summaries_root / "annotation_summary.json")
    save_table(
        pd.DataFrame([summary]),
        out.tables_root / "annotation_summary.csv",
        out.tables_root / "annotation_summary.tex",
    )

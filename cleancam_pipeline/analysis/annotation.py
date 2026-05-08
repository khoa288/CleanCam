"""Annotation agreement analysis functions."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def compute_annotation_agreement(
    annotation_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Compute inter-annotator agreement metrics.

    Args:
        annotation_df: DataFrame with columns: image_id, annotator, label
                      Optional column: resolved_label

    Returns:
        Tuple of (pairwise_agreement, first_pair_confusion, image_level, summary_dict)
    """
    needed = {"image_id", "annotator", "label"}
    missing = needed - set(annotation_df.columns)
    if missing:
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")

    pivot = annotation_df.pivot_table(
        index="image_id", columns="annotator", values="label", aggfunc="first"
    )
    annotators = list(pivot.columns)

    # Pairwise agreement
    pair_rows = []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a_name, b_name = annotators[i], annotators[j]
            pair = pivot[[a_name, b_name]].dropna()
            if pair.empty:
                continue
            y1 = pair[a_name].astype(int).to_numpy()
            y2 = pair[b_name].astype(int).to_numpy()
            pair_rows.append(
                {
                    "annotator_a": a_name,
                    "annotator_b": b_name,
                    "n_images": int(len(pair)),
                    "cohen_kappa": float(cohen_kappa_score(y1, y2)),
                    "quadratic_weighted_kappa": float(
                        cohen_kappa_score(y1, y2, weights="quadratic")
                    ),
                    "raw_agreement": float(np.mean(y1 == y2)),
                }
            )
    pair_df = pd.DataFrame(pair_rows)

    # First pair confusion matrix
    first_pair_conf = pd.DataFrame()
    if len(annotators) >= 2:
        pair = pivot[[annotators[0], annotators[1]]].dropna()
        if not pair.empty:
            from cleancam_pipeline.core.constants import LABELS

            cm = confusion_matrix(
                pair[annotators[0]].astype(int),
                pair[annotators[1]].astype(int),
                labels=LABELS,
            )
            first_pair_conf = pd.DataFrame(
                cm,
                index=[f"A_L{l}" for l in LABELS],
                columns=[f"B_L{l}" for l in LABELS],
            )

    # Image-level disagreement
    image_level = (
        annotation_df.groupby("image_id")["label"]
        .agg(lambda s: list(map(int, s.dropna())))
        .reset_index()
    )
    img_rows = []
    for _, row in image_level.iterrows():
        labels = row["label"]
        uniq = sorted(set(labels))
        img_rows.append(
            {
                "image_id": row["image_id"],
                "n_votes": len(labels),
                "n_unique_labels": len(uniq),
                "min_label": min(uniq),
                "max_label": max(uniq),
                "disagreement_span": max(uniq) - min(uniq),
                "needs_adjudication": int(len(uniq) > 1),
            }
        )
    image_df = pd.DataFrame(img_rows)

    # Summary
    adjudication_rate = (
        float(image_df["needs_adjudication"].mean()) if not image_df.empty else float("nan")
    )
    summary = {
        "n_annotators": int(annotation_df["annotator"].nunique()),
        "n_images": int(annotation_df["image_id"].nunique()),
        "mean_cohen_kappa": (
            float(pair_df["cohen_kappa"].mean()) if not pair_df.empty else float("nan")
        ),
        "mean_quadratic_weighted_kappa": (
            float(pair_df["quadratic_weighted_kappa"].mean())
            if not pair_df.empty
            else float("nan")
        ),
        "mean_raw_agreement": (
            float(pair_df["raw_agreement"].mean()) if not pair_df.empty else float("nan")
        ),
        "adjudication_rate": adjudication_rate,
        "images_with_span_gt_1": (
            int((image_df["disagreement_span"] > 1).sum()) if not image_df.empty else 0
        ),
    }

    return pair_df, first_pair_conf, image_df, summary

"""Result aggregation and comparison functions."""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cleancam_pipeline.core.constants import LABELS, PRIMARY_SETTINGS


def aggregate_benchmark_results(
    results: List[Dict[str, object]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate benchmark results across seeds.

    Args:
        results: List of result dictionaries from train_one_setting

    Returns:
        Tuple of (main_df, per_class_df, binary_df, pairwise_df)
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in results:
        grouped[(r["model_name"], r["setting_name"])].append(r)

    main_rows = []
    per_class_rows = []
    binary_rows = []

    for (model_name, setting_name), group in grouped.items():
        test_metrics = [g["test_metrics"] for g in group]
        main_rows.append(
            {
                "model": model_name,
                "setting": setting_name,
                "is_primary": int(setting_name in PRIMARY_SETTINGS),
                "n_seeds": len(group),
                "accuracy_mean": float(np.mean([m["accuracy"] for m in test_metrics])),
                "accuracy_std": float(np.std([m["accuracy"] for m in test_metrics], ddof=0)),
                "macro_f1_mean": float(np.mean([m["macro_f1"] for m in test_metrics])),
                "macro_f1_std": float(np.std([m["macro_f1"] for m in test_metrics], ddof=0)),
                "weighted_kappa_mean": float(
                    np.mean([m["weighted_kappa"] for m in test_metrics])
                ),
                "weighted_kappa_std": float(
                    np.std([m["weighted_kappa"] for m in test_metrics], ddof=0)
                ),
                "mae_mean": float(np.mean([m["mae"] for m in test_metrics])),
                "mae_std": float(np.std([m["mae"] for m in test_metrics], ddof=0)),
                "within1_mean": float(np.mean([m["within1"] for m in test_metrics])),
                "within1_std": float(np.std([m["within1"] for m in test_metrics], ddof=0)),
            }
        )

        for label in LABELS:
            label_metrics = []
            for g in group:
                row = next(
                    item
                    for item in g["test_metrics"]["per_class_rows"]
                    if int(item["label"]) == label
                )
                label_metrics.append(row)
            per_class_rows.append(
                {
                    "model": model_name,
                    "setting": setting_name,
                    "label": label,
                    "precision_mean": float(
                        np.mean([m["precision"] for m in label_metrics])
                    ),
                    "precision_std": float(
                        np.std([m["precision"] for m in label_metrics], ddof=0)
                    ),
                    "recall_mean": float(np.mean([m["recall"] for m in label_metrics])),
                    "recall_std": float(
                        np.std([m["recall"] for m in label_metrics], ddof=0)
                    ),
                    "f1_mean": float(np.mean([m["f1"] for m in label_metrics])),
                    "f1_std": float(np.std([m["f1"] for m in label_metrics], ddof=0)),
                    "support_mean": float(np.mean([m["support"] for m in label_metrics])),
                }
            )

        binary_metrics = [g["test_metrics"]["binary_metrics"] for g in group]
        binary_rows.append(
            {
                "model": model_name,
                "setting": setting_name,
                "precision_mean": float(
                    np.mean([m["precision"] for m in binary_metrics])
                ),
                "precision_std": float(
                    np.std([m["precision"] for m in binary_metrics], ddof=0)
                ),
                "recall_mean": float(np.mean([m["recall"] for m in binary_metrics])),
                "recall_std": float(
                    np.std([m["recall"] for m in binary_metrics], ddof=0)
                ),
                "f1_mean": float(np.mean([m["f1"] for m in binary_metrics])),
                "f1_std": float(np.std([m["f1"] for m in binary_metrics], ddof=0)),
                "auroc_mean": float(np.mean([m["auroc"] for m in binary_metrics])),
                "auroc_std": float(np.std([m["auroc"] for m in binary_metrics], ddof=0)),
                "auprc_mean": float(np.mean([m["auprc"] for m in binary_metrics])),
                "auprc_std": float(np.std([m["auprc"] for m in binary_metrics], ddof=0)),
            }
        )

    main_df = pd.DataFrame(main_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    binary_df = pd.DataFrame(binary_rows)

    # Pairwise improvements
    pairwise_rows = []
    pairs = [
        (
            "train_real_only__eval_real_only",
            "train_real_plus_synthetic__eval_real_only",
            "real_only",
        ),
        (
            "train_real_only__eval_real_plus_synthetic",
            "train_real_plus_synthetic__eval_real_plus_synthetic",
            "real_plus_synthetic",
        ),
    ]
    for model_name in sorted(main_df["model"].unique()):
        model_df = main_df[main_df["model"] == model_name].set_index("setting")
        for left, right, eval_domain in pairs:
            if left in model_df.index and right in model_df.index:
                pairwise_rows.append(
                    {
                        "model": model_name,
                        "eval_domain": eval_domain,
                        "delta_macro_f1": float(
                            model_df.loc[right, "macro_f1_mean"]
                            - model_df.loc[left, "macro_f1_mean"]
                        ),
                        "delta_weighted_kappa": float(
                            model_df.loc[right, "weighted_kappa_mean"]
                            - model_df.loc[left, "weighted_kappa_mean"]
                        ),
                        "delta_accuracy": float(
                            model_df.loc[right, "accuracy_mean"]
                            - model_df.loc[left, "accuracy_mean"]
                        ),
                        "delta_mae": float(
                            model_df.loc[right, "mae_mean"] - model_df.loc[left, "mae_mean"]
                        ),
                    }
                )
    pairwise_df = pd.DataFrame(pairwise_rows)

    return main_df, per_class_df, binary_df, pairwise_df

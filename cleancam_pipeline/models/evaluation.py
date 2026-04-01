"""Model evaluation utilities."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from cleancam_pipeline.core.constants import INDEX_TO_LABEL, LABEL_TO_INDEX, LABELS
from cleancam_pipeline.utils.metrics import compute_within_one_accuracy


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, object]:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        loader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    logits_all = []
    targets_all = []
    ids_all = []

    with torch.no_grad():
        for x, y, ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            logits_all.append(logits.detach().cpu())
            targets_all.append(y.detach().cpu())
            ids_all.extend(list(ids))

    logits = torch.cat(logits_all, dim=0).numpy()
    targets_idx = torch.cat(targets_all, dim=0).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds_idx = probs.argmax(axis=1)

    true_labels = np.array([INDEX_TO_LABEL[idx] for idx in targets_idx])
    pred_labels = np.array([INDEX_TO_LABEL[idx] for idx in preds_idx])

    # Main metrics
    acc = float(accuracy_score(targets_idx, preds_idx))
    macro_f1 = float(f1_score(targets_idx, preds_idx, average="macro"))
    weighted_kappa = float(
        cohen_kappa_score(true_labels, pred_labels, weights="quadratic")
    )
    mae = float(np.mean(np.abs(true_labels - pred_labels)))
    within1 = compute_within_one_accuracy(true_labels, pred_labels)

    # Per-class metrics
    prfs = precision_recall_fscore_support(
        targets_idx, preds_idx, labels=list(range(len(LABELS))), zero_division=0
    )
    per_class_rows = []
    for idx, label in INDEX_TO_LABEL.items():
        per_class_rows.append(
            {
                "label": int(label),
                "precision": float(prfs[0][idx]),
                "recall": float(prfs[1][idx]),
                "f1": float(prfs[2][idx]),
                "support": int(prfs[3][idx]),
            }
        )

    # Confusion matrices
    cm = confusion_matrix(targets_idx, preds_idx, labels=list(range(len(LABELS))))
    cm_norm = confusion_matrix(
        targets_idx, preds_idx, labels=list(range(len(LABELS))), normalize="true"
    )

    # Binary metrics (severe vs non-severe)
    y_true_bin = np.isin(true_labels, [4, 5]).astype(int)
    y_pred_bin = np.isin(pred_labels, [4, 5]).astype(int)
    y_prob_bin = probs[:, LABEL_TO_INDEX[4]] + probs[:, LABEL_TO_INDEX[5]]

    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )

    try:
        auroc = float(roc_auc_score(y_true_bin, y_prob_bin))
    except Exception:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_true_bin, y_prob_bin))
    except Exception:
        auprc = float("nan")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_kappa": weighted_kappa,
        "mae": mae,
        "within1": within1,
        "per_class_rows": per_class_rows,
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm,
        "targets_idx": targets_idx,
        "preds_idx": preds_idx,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "probs": probs,
        "image_ids": ids_all,
        "binary_metrics": {
            "precision": float(bin_p),
            "recall": float(bin_r),
            "f1": float(bin_f1),
            "auroc": auroc,
            "auprc": auprc,
        },
    }

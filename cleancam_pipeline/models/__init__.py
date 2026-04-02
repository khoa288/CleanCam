"""Model building, training, and evaluation components."""

from cleancam_pipeline.models.aggregation import aggregate_benchmark_results
from cleancam_pipeline.models.builder import build_model, compute_loss_fn
from cleancam_pipeline.models.evaluation import evaluate_model
from cleancam_pipeline.models.ordinal import (
    CoralHead,
    CornHead,
    compute_coral_loss,
    compute_corn_loss,
    get_ordinal_probabilities,
    predict_from_coral_logits,
    predict_from_corn_logits,
)
from cleancam_pipeline.models.training import train_one_setting

__all__ = [
    "aggregate_benchmark_results",
    "build_model",
    "compute_loss_fn",
    "evaluate_model",
    "train_one_setting",
    "CoralHead",
    "CornHead",
    "compute_coral_loss",
    "compute_corn_loss",
    "get_ordinal_probabilities",
    "predict_from_coral_logits",
    "predict_from_corn_logits",
]

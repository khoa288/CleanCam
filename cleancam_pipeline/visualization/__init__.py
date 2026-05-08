"""Visualization components for CleanCam pipeline."""

from cleancam_pipeline.visualization.confusion_matrix import plot_confusion_matrix
from cleancam_pipeline.visualization.plots import (
    make_example_grid,
    plot_hist_comparison,
    plot_label_distribution_overall,
    plot_pca_stats,
    plot_real_counts_by_day_camera_session,
    plot_setting_metric,
    plot_split_label_distribution,
    plot_synthetic_source_target,
)

__all__ = [
    "plot_confusion_matrix",
    "make_example_grid",
    "plot_hist_comparison",
    "plot_label_distribution_overall",
    "plot_pca_stats",
    "plot_real_counts_by_day_camera_session",
    "plot_setting_metric",
    "plot_split_label_distribution",
    "plot_synthetic_source_target",
]

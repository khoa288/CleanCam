"""Confusion matrix visualization."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray, labels: Sequence[int], title: str, out_path: Path
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix array
        labels: Label values
        title: Plot title
        out_path: Output file path
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, [f"L{l}" for l in labels])
    plt.yticks(tick_marks, [f"L{l}" for l in labels])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

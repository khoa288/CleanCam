"""Synthetic data analysis functions."""

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def grayscale_entropy(gray: np.ndarray) -> float:
    """
    Compute entropy of grayscale image.

    Args:
        gray: Grayscale image array

    Returns:
        Entropy value
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    hist = hist / max(hist.sum(), 1.0)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return float(-(hist * np.log2(hist)).sum())


def laplacian_variance(gray: np.ndarray) -> float:
    """
    Compute Laplacian variance (sharpness measure).

    Args:
        gray: Grayscale image array

    Returns:
        Laplacian variance
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    """
    Compute Tenengrad sharpness measure.

    Args:
        gray: Grayscale image array

    Returns:
        Tenengrad value
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def rms_contrast(gray: np.ndarray) -> float:
    """
    Compute RMS contrast.

    Args:
        gray: Grayscale image array

    Returns:
        RMS contrast value
    """
    return float(gray.astype(np.float64).std())


def brightness_mean(gray: np.ndarray) -> float:
    """
    Compute mean brightness.

    Args:
        gray: Grayscale image array

    Returns:
        Mean brightness value
    """
    return float(gray.astype(np.float64).mean())


def extract_low_level_stats(image_path: Path) -> Dict[str, float]:
    """
    Extract low-level image statistics.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary of statistics

    Raises:
        RuntimeError: If image cannot be read
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Failed to read image for low-level stats: {image_path}")

    return {
        "laplacian_var": laplacian_variance(gray),
        "tenengrad": tenengrad(gray),
        "rms_contrast": rms_contrast(gray),
        "entropy": grayscale_entropy(gray),
        "brightness_mean": brightness_mean(gray),
    }

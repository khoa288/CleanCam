"""Image processing utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB numpy array."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def image_to_pil(path: Path) -> Image.Image:
    """Load image as PIL Image in RGB mode."""
    return Image.open(path).convert("RGB")


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(
    path: Path, hash_size: int = 8, highfreq_factor: int = 4
) -> np.ndarray:
    """
    Compute perceptual hash of image.

    Args:
        path: Path to image file
        hash_size: Size of hash (default 8 for 64-bit hash)
        highfreq_factor: Factor for DCT size

    Returns:
        Binary hash as numpy array
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image for pHash: {path}")

    size = hash_size * highfreq_factor
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = np.float32(img)
    dct = cv2.dct(img)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:]) if low.size > 1 else np.median(low)
    return (low > med).astype(np.uint8).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between two binary arrays."""
    return int(np.count_nonzero(a != b))

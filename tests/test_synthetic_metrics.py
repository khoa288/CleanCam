from __future__ import annotations

import numpy as np

from cleancam_pipeline.analysis.synthetic import (
    brightness_mean,
    grayscale_entropy,
    laplacian_variance,
    rms_contrast,
    tenengrad,
)


def test_constant_image_metrics() -> None:
    image = np.full((8, 8), 17, dtype=np.uint8)
    assert brightness_mean(image) == 17.0
    assert rms_contrast(image) == 0.0
    assert grayscale_entropy(image) == 0.0
    assert laplacian_variance(image) == 0.0
    assert tenengrad(image) == 0.0


def test_entropy_and_population_rms() -> None:
    image = np.array([[0, 2], [0, 2]], dtype=np.uint8)
    assert grayscale_entropy(image) == 1.0
    assert brightness_mean(image) == 1.0
    assert rms_contrast(image) == 1.0

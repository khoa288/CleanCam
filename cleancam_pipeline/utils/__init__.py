"""Utility functions, imported lazily so PyTorch remains optional."""

from importlib import import_module
from typing import Any

__all__ = [
    "OutputManager",
    "ensure_dir",
    "save_csv",
    "save_json",
    "save_table",
    "compute_phash",
    "compute_sha256",
    "hamming_distance",
    "image_to_pil",
    "load_image_rgb",
    "compute_within_one_accuracy",
    "count_by_label",
    "maybe_float_series",
    "set_seed",
]

_MODULES = {
    "compute_phash": "image",
    "compute_sha256": "image",
    "hamming_distance": "image",
    "image_to_pil": "image",
    "load_image_rgb": "image",
    "OutputManager": "io",
    "ensure_dir": "io",
    "save_csv": "io",
    "save_json": "io",
    "save_table": "io",
    "compute_within_one_accuracy": "metrics",
    "count_by_label": "metrics",
    "maybe_float_series": "metrics",
    "set_seed": "seed",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(name)
    module = import_module(f"cleancam_pipeline.utils.{_MODULES[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value

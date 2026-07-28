"""Pipeline orchestrators, imported lazily to keep benchmarks optional."""

from importlib import import_module
from typing import Any

__all__ = [
    "run_annotation",
    "run_benchmark",
    "run_characterization",
    "run_integrity",
    "run_synthetic_analysis",
]

_MODULES = {
    "run_annotation": "annotation_runner",
    "run_benchmark": "benchmark_runner",
    "run_characterization": "characterization_runner",
    "run_integrity": "integrity_runner",
    "run_synthetic_analysis": "synthetic_runner",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(name)
    module = import_module(f"cleancam_pipeline.orchestrators.{_MODULES[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value

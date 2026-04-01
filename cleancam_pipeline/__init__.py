"""
CleanCam Dataset Paper Pipeline

A modular pipeline for dataset characterization, integrity auditing,
annotation agreement analysis, synthetic data analysis, and CNN benchmarking.
"""

__version__ = "2.0.0"

from cleancam_pipeline.core.config import BenchmarkConfig
from cleancam_pipeline.core.release import CleanCamRelease

__all__ = ["BenchmarkConfig", "CleanCamRelease"]

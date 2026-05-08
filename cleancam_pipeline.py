#!/usr/bin/env python3
"""
CleanCam dataset-paper analysis pipeline (Fully Refactored).

This is the complete refactored version using only modular components.
No dependencies on the original monolithic script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from cleancam_pipeline.core import BenchmarkConfig, CleanCamRelease
from cleancam_pipeline.orchestrators import (
    run_annotation,
    run_benchmark,
    run_characterization,
    run_integrity,
    run_synthetic_analysis,
)
from cleancam_pipeline.utils import OutputManager, save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CleanCam dataset-paper analysis pipeline (Fully Refactored)"
    )
    parser.add_argument(
        "--release-root", type=str, required=True, help="Path to CleanCam_release"
    )
    parser.add_argument(
        "--output-root", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--annotation-csv",
        type=str,
        default=None,
        help="Optional annotation-agreement CSV",
    )

    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-characterization", action="store_true")
    parser.add_argument("--run-integrity", action="store_true")
    parser.add_argument("--run-annotation", action="store_true")
    parser.add_argument("--run-synthetic-analysis", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")

    parser.add_argument("--run-near-duplicate-audit", action="store_true")
    parser.add_argument("--near-duplicate-cap", type=int, default=1000)
    parser.add_argument("--synthetic-analysis-cap", type=int, default=2000)

    parser.add_argument(
        "--models",
        nargs="+",
        default=["mobilenet_v2", "resnet18", "efficientnet_b0"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--ordinal-methods",
        nargs="+",
        choices=["coral", "corn"],
        default=None,
        help="Ordinal regression methods (CORAL and/or CORN). If not specified, uses standard cross-entropy. Can specify multiple: --ordinal-methods coral corn",
    )
    parser.add_argument(
        "--benchmark-settings",
        nargs="+",
        choices=[
            "train_real_only__eval_real_only",
            "train_real_only__eval_real_plus_synthetic",
            "train_real_plus_synthetic__eval_real_only",
            "train_real_plus_synthetic__eval_real_plus_synthetic",
        ],
        default=None,
        help="Benchmark settings to run. If not specified, runs all 4 settings. Can specify multiple.",
    )
    parser.add_argument("--disable-weighted-sampler", action="store_true")
    parser.add_argument("--enable-class-weights", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--single-gpu", action="store_true", help="Disable multi-GPU training (use only 1 GPU)")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cleancam-dataset-paper")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-run-prefix", type=str, default="cleancam")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    release = CleanCamRelease(Path(args.release_root))
    out = OutputManager(Path(args.output_root))

    if args.run_all:
        args.run_characterization = True
        args.run_integrity = True
        args.run_synthetic_analysis = True
        args.run_benchmark = True
        if args.annotation_csv is not None:
            args.run_annotation = True

    cfg = BenchmarkConfig.from_args(args)

    env_summary = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        ),
        "benchmark_config": cfg.__dict__,
    }
    save_json(env_summary, out.summaries_root / "environment_summary.json")

    if args.run_characterization:
        print("[Stage] characterization", flush=True)
        run_characterization(release, out)

    if args.run_integrity:
        print("[Stage] integrity", flush=True)
        run_integrity(
            release,
            out,
            run_near_duplicate_audit=args.run_near_duplicate_audit,
            near_duplicate_cap=args.near_duplicate_cap,
        )

    if args.run_annotation:
        if args.annotation_csv is None:
            raise ValueError("--run-annotation requires --annotation-csv")
        print("[Stage] annotation", flush=True)
        run_annotation(Path(args.annotation_csv), out)

    if args.run_synthetic_analysis:
        print("[Stage] synthetic-analysis", flush=True)
        run_synthetic_analysis(release, out, cap=args.synthetic_analysis_cap)

    if args.run_benchmark:
        print("[Stage] benchmark", flush=True)
        run_benchmark(release, out, cfg)

    print(f"Done. Outputs written to: {out.output_root}", flush=True)


if __name__ == "__main__":
    main()

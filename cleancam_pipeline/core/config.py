"""Configuration classes for CleanCam pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BenchmarkConfig:
    """Configuration for CNN benchmark training."""

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seeds: Tuple[int, ...] = (42, 43, 44)
    models: Tuple[str, ...] = ("mobilenet_v2", "resnet18", "efficientnet_b0")
    ordinal_methods: Tuple[Optional[str], ...] = (None,)  # Tuple of None, 'coral', or 'corn'
    use_weighted_sampler: bool = True
    use_class_weights: bool = False
    train_on_gpu_if_available: bool = True
    save_best_checkpoints: bool = True
    use_amp: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    log_interval: int = 25
    use_wandb: bool = False
    wandb_project: str = "cleancam-dataset-paper"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"
    wandb_run_prefix: str = "cleancam"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not self.models:
            raise ValueError("models tuple cannot be empty")
        if not self.seeds:
            raise ValueError("seeds tuple cannot be empty")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_mode must be 'online', 'offline', or 'disabled'")
        if not self.ordinal_methods:
            raise ValueError("ordinal_methods tuple cannot be empty")
        for method in self.ordinal_methods:
            if method not in (None, "coral", "corn"):
                raise ValueError(f"ordinal_method must be None, 'coral', or 'corn', got: {method}")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BenchmarkConfig:
        """Create configuration from command-line arguments."""
        # Handle ordinal_methods - convert list to tuple, handle None
        ordinal_methods = tuple(args.ordinal_methods) if hasattr(args, 'ordinal_methods') and args.ordinal_methods else (None,)
        
        return cls(
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seeds=tuple(args.seeds),
            models=tuple(args.models),
            ordinal_methods=ordinal_methods,
            use_weighted_sampler=not args.disable_weighted_sampler,
            use_class_weights=args.enable_class_weights,
            train_on_gpu_if_available=not args.cpu_only,
            save_best_checkpoints=not args.no_save_checkpoints,
            use_amp=not args.no_amp,
            persistent_workers=not args.no_persistent_workers,
            prefetch_factor=args.prefetch_factor,
            log_interval=args.log_interval,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
            wandb_run_prefix=args.wandb_run_prefix,
        )

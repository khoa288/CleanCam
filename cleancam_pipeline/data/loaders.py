"""DataLoader creation utilities."""

from collections import Counter
from typing import Dict

import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from cleancam_pipeline.core.config import BenchmarkConfig
from cleancam_pipeline.core.constants import LABEL_TO_INDEX
from cleancam_pipeline.data.dataset import CleanCamClassificationDataset


def loader_kwargs(cfg: BenchmarkConfig) -> Dict[str, object]:
    """Build common DataLoader kwargs from config."""
    kwargs: Dict[str, object] = {"num_workers": cfg.num_workers, "pin_memory": True}
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return kwargs


def make_train_loader(
    df: pd.DataFrame, transform: transforms.Compose, cfg: BenchmarkConfig
) -> DataLoader:
    """
    Create training DataLoader with optional weighted sampling.

    Args:
        df: Training data DataFrame
        transform: Training transforms
        cfg: Benchmark configuration

    Returns:
        Training DataLoader
    """
    dataset = CleanCamClassificationDataset(df, transform)

    if cfg.use_weighted_sampler:
        label_indices = [LABEL_TO_INDEX[int(label)] for label in df["label"].tolist()]
        counts = Counter(label_indices)
        weights = [1.0 / counts[idx] for idx in label_indices]
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
        return DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=sampler, **loader_kwargs(cfg)
        )

    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs(cfg)
    )


def make_eval_loader(
    df: pd.DataFrame, transform: transforms.Compose, cfg: BenchmarkConfig
) -> DataLoader:
    """
    Create evaluation DataLoader.

    Args:
        df: Evaluation data DataFrame
        transform: Evaluation transforms
        cfg: Benchmark configuration

    Returns:
        Evaluation DataLoader
    """
    dataset = CleanCamClassificationDataset(df, transform)
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs(cfg)
    )

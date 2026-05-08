"""CleanCam release data loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def find_existing_split_path(split_root: Path, candidates: Sequence[str]) -> Path:
    """Find first existing split file from candidates."""
    for name in candidates:
        path = split_root / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of the candidate split files exist under {split_root}: {candidates}"
    )


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


class CleanCamRelease:
    """Manages access to CleanCam release data and splits."""

    def __init__(self, release_root: Path):
        """
        Initialize CleanCam release loader.

        Args:
            release_root: Path to CleanCam_release directory
        """
        self.release_root = Path(release_root)
        self.metadata_root = self.release_root / "metadata"
        self.splits_root = self.release_root / "splits" / "official"
        self.images_root = self.release_root / "images"

        # Load metadata
        self.metadata = read_csv(self.metadata_root / "metadata.csv")
        self.metadata_real = read_csv(self.metadata_root / "metadata_real.csv")
        self.metadata_synth = read_csv(self.metadata_root / "metadata_synthetic.csv")

        # Load build summary
        build_summary_path = self.metadata_root / "build_summary.json"
        self.build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))

        # Load split summary if available
        split_summary_path = self.metadata_root / "split_summary.csv"
        if split_summary_path.exists():
            self.split_summary = read_csv(split_summary_path)
        else:
            self.split_summary = pd.DataFrame()

        # Load official splits
        self.official_splits = self._load_official_splits()

        # Prepare absolute paths
        self._prepare_paths()

        # Build benchmark settings
        self.settings = self._build_settings()

    def _load_official_splits(self) -> Dict[str, pd.DataFrame]:
        """Load all official split files."""
        return {
            "train_real_only": read_csv(
                find_existing_split_path(self.splits_root, ["train_real_only.csv"])
            ),
            "train_real_plus_synthetic": read_csv(
                find_existing_split_path(
                    self.splits_root, ["train_real_plus_synthetic.csv"]
                )
            ),
            "val_real_only": read_csv(
                find_existing_split_path(
                    self.splits_root, ["val_real_only.csv", "val.csv"]
                )
            ),
            "val_real_plus_synthetic": read_csv(
                find_existing_split_path(
                    self.splits_root, ["val_real_plus_synthetic.csv"]
                )
            ),
            "test_real_only": read_csv(
                find_existing_split_path(
                    self.splits_root, ["test_real_only.csv", "test.csv"]
                )
            ),
            "test_real_plus_synthetic": read_csv(
                find_existing_split_path(
                    self.splits_root, ["test_real_plus_synthetic.csv"]
                )
            ),
        }

    def _prepare_paths(self) -> None:
        """Add absolute_path column to all dataframes."""
        dfs = [
            self.metadata,
            self.metadata_real,
            self.metadata_synth,
        ] + list(self.official_splits.values())

        for df in dfs:
            if "relative_path" in df.columns:
                df["absolute_path"] = df["relative_path"].map(
                    lambda rp: str(self.release_root / str(rp))
                )
            elif "release_path" in df.columns:
                df["absolute_path"] = df["release_path"]
            else:
                raise KeyError("Expected relative_path or release_path in dataframe")

    def _build_settings(self) -> Dict[str, Dict]:
        """Build benchmark setting configurations."""
        return {
            "train_real_only__eval_real_only": {
                "train": self.official_splits["train_real_only"],
                "val": self.official_splits["val_real_only"],
                "test": self.official_splits["test_real_only"],
                "train_name": "train_real_only",
                "eval_name": "real_only",
            },
            "train_real_plus_synthetic__eval_real_only": {
                "train": self.official_splits["train_real_plus_synthetic"],
                "val": self.official_splits["val_real_only"],
                "test": self.official_splits["test_real_only"],
                "train_name": "train_real_plus_synthetic",
                "eval_name": "real_only",
            },
            "train_real_only__eval_real_plus_synthetic": {
                "train": self.official_splits["train_real_only"],
                "val": self.official_splits["val_real_plus_synthetic"],
                "test": self.official_splits["test_real_plus_synthetic"],
                "train_name": "train_real_only",
                "eval_name": "real_plus_synthetic",
            },
            "train_real_plus_synthetic__eval_real_plus_synthetic": {
                "train": self.official_splits["train_real_plus_synthetic"],
                "val": self.official_splits["val_real_plus_synthetic"],
                "test": self.official_splits["test_real_plus_synthetic"],
                "train_name": "train_real_plus_synthetic",
                "eval_name": "real_plus_synthetic",
            },
        }

    def get_split(self, split_name: str) -> pd.DataFrame:
        """Get a specific split by name."""
        if split_name not in self.official_splits:
            raise ValueError(
                f"Unknown split: {split_name}. "
                f"Available: {list(self.official_splits.keys())}"
            )
        return self.official_splits[split_name]

    def get_setting(self, setting_name: str) -> Dict:
        """Get a specific benchmark setting by name."""
        if setting_name not in self.settings:
            raise ValueError(
                f"Unknown setting: {setting_name}. "
                f"Available: {list(self.settings.keys())}"
            )
        return self.settings[setting_name]

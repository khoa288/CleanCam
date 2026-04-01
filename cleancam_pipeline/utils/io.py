"""File I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(obj: dict, path: Path) -> None:
    """Save dictionary to JSON file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_table(
    df: pd.DataFrame,
    csv_path: Path,
    latex_path: Optional[Path] = None,
    index: bool = False,
) -> None:
    """Save DataFrame as CSV and optionally as LaTeX table."""
    save_csv(df, csv_path)
    if latex_path is not None:
        ensure_dir(latex_path.parent)
        latex = df.to_latex(index=index, escape=False)
        latex_path.write_text(latex, encoding="utf-8")


class OutputManager:
    """Manages output directory structure for pipeline results."""

    def __init__(self, output_root: Path):
        """
        Initialize output manager.

        Args:
            output_root: Root directory for all outputs
        """
        self.output_root = Path(output_root)
        self.tables_root = self.output_root / "tables"
        self.figures_root = self.output_root / "figures"
        self.logs_root = self.output_root / "logs"
        self.summaries_root = self.output_root / "summaries"
        self.benchmark_root = self.output_root / "benchmark"

        # Create all directories
        for directory in [
            self.tables_root,
            self.figures_root,
            self.logs_root,
            self.summaries_root,
            self.benchmark_root,
        ]:
            ensure_dir(directory)

    def get_benchmark_output_dir(
        self, model_name: str, setting_name: str, seed: int
    ) -> Path:
        """Get output directory for a specific benchmark run."""
        path = self.benchmark_root / model_name / setting_name / f"seed_{seed}"
        ensure_dir(path)
        return path

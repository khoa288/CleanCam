from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_help_does_not_require_optional_benchmark_packages() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "cleancam_pipeline.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "optional benchmark pipeline" in completed.stdout

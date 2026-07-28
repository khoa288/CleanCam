from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_examples_are_unique_and_well_formed() -> None:
    config = json.loads(
        (ROOT / "configs/manuscript_examples.json").read_text(encoding="utf-8")
    )
    assert set(config) == {"figure_2_real", "figure_4_synthetic"}
    assert set(config["figure_2_real"]) == {"1", "2", "3", "4", "5"}
    assert set(config["figure_4_synthetic"]) == {"3", "4", "5"}
    identifiers = [
        image_id
        for figure in config.values()
        for label_ids in figure.values()
        for image_id in label_ids
    ]
    assert len(identifiers) == 24
    assert len(set(identifiers)) == 24
    assert all(len(label_ids) == 3 for figure in config.values() for label_ids in figure.values())


def test_documentation_csv_files_parse() -> None:
    for filename in ("data_dictionary.csv", "deposit_assets.csv", "label_taxonomy.csv"):
        frame = pd.read_csv(ROOT / "docs" / filename)
        assert not frame.empty
        assert frame.columns.is_unique

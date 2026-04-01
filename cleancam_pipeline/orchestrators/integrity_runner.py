"""Integrity audit orchestrator."""

from typing import Optional

import pandas as pd

from cleancam_pipeline.analysis.integrity import (
    audit_exact_duplicates,
    audit_integrity,
    audit_near_duplicates_between_splits,
)
from cleancam_pipeline.core.release import CleanCamRelease
from cleancam_pipeline.utils.io import OutputManager, save_json, save_table


def run_integrity(
    release: CleanCamRelease,
    out: OutputManager,
    run_near_duplicate_audit: bool,
    near_duplicate_cap: Optional[int],
) -> None:
    """
    Run complete integrity audit.

    Args:
        release: CleanCam release data
        out: Output manager
        run_near_duplicate_audit: Whether to run near-duplicate checks
        near_duplicate_cap: Maximum images to check per split
    """
    # Run integrity checks
    integrity = audit_integrity(
        release.official_splits["train_real_only"],
        release.official_splits["val_real_only"],
        release.official_splits["test_real_only"],
        release.official_splits["train_real_plus_synthetic"],
        release.official_splits["val_real_plus_synthetic"],
        release.official_splits["test_real_plus_synthetic"],
        release.metadata_real,
    )
    exact_dups = audit_exact_duplicates(release.metadata)

    save_table(
        integrity,
        out.tables_root / "integrity_audit.csv",
        out.tables_root / "integrity_audit.tex",
    )
    save_table(exact_dups, out.tables_root / "exact_duplicate_audit.csv")

    summary = {
        "integrity_checks": integrity.to_dict(orient="records"),
        "exact_duplicate_rows": exact_dups.to_dict(orient="records"),
    }

    if run_near_duplicate_audit:
        pairs = [
            ("train_real_only", "val_real_only"),
            ("train_real_only", "test_real_only"),
            ("val_real_only", "test_real_only"),
        ]
        near_dup_counts = []
        for a_name, b_name in pairs:
            df = audit_near_duplicates_between_splits(
                release.official_splits[a_name],
                release.official_splits[b_name],
                cap=near_duplicate_cap,
            )
            save_table(
                df, out.tables_root / f"near_duplicate_{a_name}__{b_name}.csv"
            )
            near_dup_counts.append({"pair": f"{a_name}__{b_name}", "n_pairs": int(len(df))})
        save_table(
            pd.DataFrame(near_dup_counts),
            out.tables_root / "near_duplicate_summary.csv",
        )
        summary["near_duplicate_summary"] = near_dup_counts

    save_json(summary, out.summaries_root / "integrity_summary.json")

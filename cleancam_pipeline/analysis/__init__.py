"""Analysis components for CleanCam dataset."""

from cleancam_pipeline.analysis.annotation import compute_annotation_agreement
from cleancam_pipeline.analysis.characterization import (
    make_group_count_table,
    make_release_composition_table,
    make_split_composition_table,
)
from cleancam_pipeline.analysis.integrity import (
    audit_exact_duplicates,
    audit_integrity,
    audit_near_duplicates_between_splits,
)
from cleancam_pipeline.analysis.synthetic import (
    extract_low_level_stats,
    grayscale_entropy,
    laplacian_variance,
)

__all__ = [
    "compute_annotation_agreement",
    "make_group_count_table",
    "make_release_composition_table",
    "make_split_composition_table",
    "audit_exact_duplicates",
    "audit_integrity",
    "audit_near_duplicates_between_splits",
    "extract_low_level_stats",
    "grayscale_entropy",
    "laplacian_variance",
]

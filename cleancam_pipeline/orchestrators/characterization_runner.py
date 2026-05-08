"""Characterization analysis orchestrator."""

from cleancam_pipeline.analysis.characterization import (
    make_group_count_table,
    make_release_composition_table,
    make_split_composition_table,
)
from cleancam_pipeline.core.release import CleanCamRelease
from cleancam_pipeline.utils.io import OutputManager, save_json, save_table
from cleancam_pipeline.visualization.plots import (
    make_example_grid,
    plot_label_distribution_overall,
    plot_real_counts_by_day_camera_session,
    plot_split_label_distribution,
    plot_synthetic_source_target,
)


def run_characterization(release: CleanCamRelease, out: OutputManager) -> None:
    """
    Run complete characterization analysis.

    Args:
        release: CleanCam release data
        out: Output manager
    """
    # Generate tables
    release_comp = make_release_composition_table(
        release.metadata, release.metadata_real, release.metadata_synth
    )
    split_comp = make_split_composition_table(release.official_splits)
    counts_by_cam = make_group_count_table(release.metadata_real, "cam")
    counts_by_state = make_group_count_table(release.metadata_real, "state")
    counts_by_day = make_group_count_table(release.metadata_real, "day")
    counts_by_capture = make_group_count_table(release.metadata_real, "capture_id")

    synth_parent = (
        release.metadata_synth.groupby("source_label")
        .size()
        .reset_index(name="n_images")
        if not release.metadata_synth.empty
        else None
    )
    synth_src_tgt = (
        release.metadata_synth.groupby(["source_label", "label"])
        .size()
        .reset_index(name="n_images")
        if not release.metadata_synth.empty
        else None
    )

    # Save tables
    save_table(
        release_comp,
        out.tables_root / "release_composition.csv",
        out.tables_root / "release_composition.tex",
    )
    save_table(
        split_comp,
        out.tables_root / "official_split_composition.csv",
        out.tables_root / "official_split_composition.tex",
    )
    save_table(counts_by_cam, out.tables_root / "counts_by_camera.csv")
    save_table(counts_by_state, out.tables_root / "counts_by_session.csv")
    save_table(counts_by_day, out.tables_root / "counts_by_day.csv")
    save_table(counts_by_capture, out.tables_root / "counts_by_capture.csv")

    if synth_parent is not None:
        save_table(
            synth_parent, out.tables_root / "synthetic_parent_label_distribution.csv"
        )
    if synth_src_tgt is not None:
        save_table(
            synth_src_tgt, out.tables_root / "synthetic_source_target_distribution.csv"
        )

    # Generate plots
    plot_label_distribution_overall(
        release.metadata,
        release.metadata_real,
        release.metadata_synth,
        out.figures_root / "label_distribution_overall.png",
    )
    plot_split_label_distribution(
        release.official_splits,
        out.figures_root / "label_distribution_official_splits.png",
    )
    plot_real_counts_by_day_camera_session(
        release.metadata_real,
        out.figures_root / "real_counts_by_day_camera_session.png",
    )
    make_example_grid(
        release.metadata, out.figures_root / "example_grid_real.png", origin="real"
    )
    make_example_grid(
        release.metadata,
        out.figures_root / "example_grid_synthetic.png",
        origin="synthetic",
    )
    plot_synthetic_source_target(
        release.metadata_synth,
        out.figures_root / "synthetic_source_target_counts.png",
    )

    # Save summary
    save_json(
        {
            "release_composition": release_comp.to_dict(orient="records"),
            "official_split_composition": split_comp.to_dict(orient="records"),
        },
        out.summaries_root / "characterization_summary.json",
    )

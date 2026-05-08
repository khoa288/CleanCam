"""Synthetic data analysis orchestrator."""

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from scipy.stats import wasserstein_distance
except Exception:  # pragma: no cover
    wasserstein_distance = None

from cleancam_pipeline.analysis.synthetic import extract_low_level_stats
from cleancam_pipeline.core.release import CleanCamRelease
from cleancam_pipeline.utils.io import OutputManager, save_json, save_table
from cleancam_pipeline.utils.metrics import count_by_label, maybe_float_series
from cleancam_pipeline.visualization.plots import plot_hist_comparison, plot_pca_stats


def run_synthetic_analysis(
    release: CleanCamRelease, out: OutputManager, cap: Optional[int]
) -> None:
    """
    Run synthetic data analysis.

    Args:
        release: CleanCam release data
        out: Output manager
        cap: Maximum images to analyze (None for all)
    """
    synth = release.metadata_synth.copy()
    real = release.metadata_real.copy()

    if synth.empty:
        save_json(
            {"message": "No synthetic rows found."},
            out.summaries_root / "synthetic_summary.json",
        )
        return

    # Parameter summary
    param_cols = [
        "rotation_deg",
        "coverage_scale",
        "blur_scale_factor",
        "opacity",
        "avg_blockage",
        "label_boost",
    ]
    param_rows = []
    for col in param_cols:
        vals = maybe_float_series(synth, col)
        if vals.empty:
            continue
        param_rows.append(
            {
                "field": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "median": float(vals.median()),
                "max": float(vals.max()),
            }
        )
    param_df = pd.DataFrame(param_rows)
    save_table(
        param_df,
        out.tables_root / "synthetic_parameter_summary.csv",
        out.tables_root / "synthetic_parameter_summary.tex",
    )

    # Severe real vs severe synthetic stats
    real_severe = real[real["label"].isin([4, 5])].copy()
    synth_severe = synth[synth["label"].isin([4, 5])].copy()
    if cap is not None:
        real_severe = real_severe.head(cap)
        synth_severe = synth_severe.head(cap)

    def collect_stats(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            try:
                stats = extract_low_level_stats(Path(row["absolute_path"]))
            except Exception:
                continue
            stats.update(
                {"group": group_name, "image_id": row["image_id"], "label": int(row["label"])}
            )
            rows.append(stats)
        return pd.DataFrame(rows)

    real_stats = collect_stats(real_severe, "real_severe")
    synth_stats = collect_stats(synth_severe, "synthetic_severe")
    combined_stats = pd.concat([real_stats, synth_stats], ignore_index=True)
    save_table(
        combined_stats, out.tables_root / "synthetic_real_severe_low_level_stats.csv"
    )

    summary_rows = []
    for metric in ["laplacian_var", "tenengrad", "rms_contrast", "entropy", "brightness_mean"]:
        row = {"metric": metric}
        if not real_stats.empty:
            vals = real_stats[metric].dropna()
            row["real_mean"] = float(vals.mean())
            row["real_std"] = float(vals.std(ddof=0))
        if not synth_stats.empty:
            vals = synth_stats[metric].dropna()
            row["synthetic_mean"] = float(vals.mean())
            row["synthetic_std"] = float(vals.std(ddof=0))
        if (
            wasserstein_distance is not None
            and not real_stats.empty
            and not synth_stats.empty
        ):
            row["wasserstein"] = float(
                wasserstein_distance(
                    real_stats[metric].dropna(), synth_stats[metric].dropna()
                )
            )
        summary_rows.append(row)
    severe_summary_df = pd.DataFrame(summary_rows)
    save_table(
        severe_summary_df,
        out.tables_root / "synthetic_real_severe_summary.csv",
        out.tables_root / "synthetic_real_severe_summary.tex",
    )

    plot_hist_comparison(
        real_stats,
        synth_stats,
        out.figures_root / "synthetic_vs_real_severe_histograms.png",
    )
    plot_pca_stats(
        real_stats, synth_stats, out.figures_root / "synthetic_vs_real_severe_pca.png"
    )

    # Parent-child deltas
    real_lookup = release.metadata_real.set_index("image_id")
    delta_rows = []
    delta_cap_df = synth.head(cap) if cap is not None else synth
    for _, row in delta_cap_df.iterrows():
        parent_id = str(row.get("parent_image_id"))
        if parent_id not in real_lookup.index:
            continue
        try:
            parent_stats = extract_low_level_stats(
                Path(real_lookup.loc[parent_id, "absolute_path"])
            )
            child_stats = extract_low_level_stats(Path(row["absolute_path"]))
        except Exception:
            continue
        delta = {
            "image_id": row["image_id"],
            "parent_image_id": parent_id,
            "source_label": int(row["source_label"]),
            "target_label": int(row["label"]),
        }
        for metric in ["laplacian_var", "tenengrad", "rms_contrast", "entropy", "brightness_mean"]:
            delta[f"parent_{metric}"] = float(parent_stats[metric])
            delta[f"child_{metric}"] = float(child_stats[metric])
            delta[f"delta_{metric}"] = float(child_stats[metric] - parent_stats[metric])
        if "avg_blockage" in row and not pd.isna(row["avg_blockage"]):
            delta["avg_blockage"] = float(row["avg_blockage"])
        delta_rows.append(delta)
    delta_df = pd.DataFrame(delta_rows)
    save_table(delta_df, out.tables_root / "synthetic_parent_child_deltas.csv")

    delta_summary_rows = []
    if not delta_df.empty:
        for metric in ["laplacian_var", "tenengrad", "rms_contrast", "entropy", "brightness_mean"]:
            vals = delta_df[f"delta_{metric}"].dropna()
            delta_summary_rows.append(
                {
                    "metric": metric,
                    "mean_delta": float(vals.mean()),
                    "std_delta": float(vals.std(ddof=0)),
                    "median_delta": float(vals.median()),
                    "min_delta": float(vals.min()),
                    "max_delta": float(vals.max()),
                }
            )
    delta_summary_df = pd.DataFrame(delta_summary_rows)
    save_table(
        delta_summary_df,
        out.tables_root / "synthetic_parent_child_delta_summary.csv",
        out.tables_root / "synthetic_parent_child_delta_summary.tex",
    )

    save_json(
        {
            "n_synthetic_total": int(len(synth)),
            "synthetic_label_counts": count_by_label(synth),
            "n_real_severe_used": int(len(real_stats)),
            "n_synthetic_severe_used": int(len(synth_stats)),
            "n_parent_child_pairs_used": int(len(delta_df)),
        },
        out.summaries_root / "synthetic_summary.json",
    )

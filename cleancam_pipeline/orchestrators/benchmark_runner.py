"""Benchmark training orchestrator."""

from dataclasses import asdict

import numpy as np
import pandas as pd

from cleancam_pipeline.core.config import BenchmarkConfig
from cleancam_pipeline.core.constants import ALL_SETTINGS, LABELS, PRIMARY_SETTINGS
from cleancam_pipeline.core.release import CleanCamRelease
from cleancam_pipeline.models.aggregation import aggregate_benchmark_results
from cleancam_pipeline.models.training import train_one_setting
from cleancam_pipeline.utils.io import OutputManager, save_json, save_table
from cleancam_pipeline.visualization.confusion_matrix import plot_confusion_matrix
from cleancam_pipeline.visualization.plots import plot_setting_metric


def run_benchmark(
    release: CleanCamRelease, out: OutputManager, cfg: BenchmarkConfig
) -> None:
    """
    Run complete benchmark training.

    Args:
        release: CleanCam release data
        out: Output manager
        cfg: Benchmark configuration
    """
    run_results = []
    manifest_rows = []

    for setting_name in ALL_SETTINGS:
        setting = release.settings[setting_name]
        train_df = setting["train"].copy()
        val_df = setting["val"].copy()
        test_df = setting["test"].copy()

        print(
            f"[BenchmarkSetup] setting={setting_name} train={len(train_df)} val={len(val_df)} test={len(test_df)}",
            flush=True,
        )

        manifest_rows.append(
            {
                "setting": setting_name,
                "train_n": int(len(train_df)),
                "val_n": int(len(val_df)),
                "test_n": int(len(test_df)),
                "train_real": int((train_df["origin"] == "real").sum()),
                "train_synthetic": int((train_df["origin"] == "synthetic").sum()),
                "val_real": int((val_df["origin"] == "real").sum()),
                "val_synthetic": int((val_df["origin"] == "synthetic").sum()),
                "test_real": int((test_df["origin"] == "real").sum()),
                "test_synthetic": int((test_df["origin"] == "synthetic").sum()),
            }
        )

        for model_name in cfg.models:
            for ordinal_method in cfg.ordinal_methods:
                for seed in cfg.seeds:
                    # Create a temporary config with single ordinal_method for this run
                    run_cfg = BenchmarkConfig(
                        image_size=cfg.image_size,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        epochs=cfg.epochs,
                        patience=cfg.patience,
                        learning_rate=cfg.learning_rate,
                        weight_decay=cfg.weight_decay,
                        seeds=(seed,),
                        models=(model_name,),
                        ordinal_methods=(ordinal_method,),
                        use_weighted_sampler=cfg.use_weighted_sampler,
                        use_class_weights=cfg.use_class_weights,
                        train_on_gpu_if_available=cfg.train_on_gpu_if_available,
                        use_multi_gpu=cfg.use_multi_gpu,
                        save_best_checkpoints=cfg.save_best_checkpoints,
                        use_amp=cfg.use_amp,
                        persistent_workers=cfg.persistent_workers,
                        prefetch_factor=cfg.prefetch_factor,
                        log_interval=cfg.log_interval,
                        use_wandb=cfg.use_wandb,
                        wandb_project=cfg.wandb_project,
                        wandb_entity=cfg.wandb_entity,
                        wandb_mode=cfg.wandb_mode,
                        wandb_run_prefix=cfg.wandb_run_prefix,
                    )
                    
                    # Create output directory with ordinal method suffix
                    method_suffix = f"_{ordinal_method}" if ordinal_method else ""
                    seed_dir = out.get_benchmark_output_dir(
                        f"{model_name}{method_suffix}", setting_name, seed
                    )
                    
                    result = train_one_setting(
                        train_df, val_df, test_df, model_name, setting_name, seed, run_cfg, seed_dir
                    )
                    # Add ordinal_method to result for tracking
                    result["ordinal_method"] = ordinal_method
                    run_results.append(result)

    manifest_df = pd.DataFrame(manifest_rows)
    save_table(manifest_df, out.tables_root / "benchmark_setting_manifest.csv")

    main_df, per_class_df, binary_df, pairwise_df = aggregate_benchmark_results(
        run_results
    )
    save_table(
        main_df,
        out.tables_root / "benchmark_summary_main.csv",
        out.tables_root / "benchmark_summary_main.tex",
    )
    save_table(per_class_df, out.tables_root / "benchmark_summary_per_class.csv")
    save_table(
        binary_df,
        out.tables_root / "benchmark_summary_binary.csv",
        out.tables_root / "benchmark_summary_binary.tex",
    )
    save_table(
        pairwise_df,
        out.tables_root / "benchmark_improvement_summary.csv",
        out.tables_root / "benchmark_improvement_summary.tex",
    )

    plot_setting_metric(
        main_df,
        "macro_f1_mean",
        out.figures_root / "benchmark_macro_f1.png",
        "Benchmark comparison: macro-F1",
    )
    plot_setting_metric(
        main_df,
        "weighted_kappa_mean",
        out.figures_root / "benchmark_weighted_kappa.png",
        "Benchmark comparison: quadratic-weighted kappa",
    )
    plot_setting_metric(
        main_df,
        "mae_mean",
        out.figures_root / "benchmark_mae.png",
        "Benchmark comparison: mean absolute error",
    )

    # Mean confusion matrices for best primary setting per model
    primary_df = main_df[main_df["setting"].isin(PRIMARY_SETTINGS)].copy()
    best_rows = (
        primary_df.sort_values(["model", "macro_f1_mean"], ascending=[True, False])
        .groupby("model")
        .head(1)
    )
    best_lookup = {(row["model"], row["setting"]) for _, row in best_rows.iterrows()}
    for model_name, setting_name in best_lookup:
        selected = [
            r
            for r in run_results
            if r["model_name"] == model_name and r["setting_name"] == setting_name
        ]
        cms = np.stack(
            [r["test_metrics"]["confusion_matrix_norm"] for r in selected], axis=0
        )
        mean_cm = cms.mean(axis=0)
        plot_confusion_matrix(
            mean_cm,
            LABELS,
            f"{model_name} | {setting_name} | mean test confusion",
            out.figures_root / f"confusion_mean_{model_name}_{setting_name}.png",
        )

    save_json(
        {
            "benchmark_config": asdict(cfg),
            "setting_manifest": manifest_df.to_dict(orient="records"),
            "main_rows": main_df.to_dict(orient="records"),
            "binary_rows": binary_df.to_dict(orient="records"),
            "improvement_rows": pairwise_df.to_dict(orient="records"),
        },
        out.summaries_root / "benchmark_summary.json",
    )

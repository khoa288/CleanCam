"""Model training utilities."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cleancam_pipeline.core.config import BenchmarkConfig
from cleancam_pipeline.data.loaders import make_eval_loader, make_train_loader
from cleancam_pipeline.data.transforms import build_transforms
from cleancam_pipeline.models.builder import build_model, compute_loss_fn
from cleancam_pipeline.models.evaluation import evaluate_model
from cleancam_pipeline.models.ordinal import compute_coral_loss, compute_corn_loss
from cleancam_pipeline.utils.io import save_table
from cleancam_pipeline.utils.metrics import format_label_distribution
from cleancam_pipeline.utils.seed import set_seed

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def autocast_context(device: torch.device, enabled: bool):
    """Create autocast context for mixed precision training."""
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    """Create gradient scaler for mixed precision training."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def maybe_init_wandb(
    cfg: BenchmarkConfig,
    model_name: str,
    setting_name: str,
    seed: int,
    output_dir: Path,
):
    """Initialize Weights & Biases logging if enabled."""
    if not cfg.use_wandb:
        return None
    if wandb is None:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` or disable --use-wandb."
        )
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
        name=f"{cfg.wandb_run_prefix}-{model_name}-{setting_name}-seed{seed}",
        group=f"{model_name}-{setting_name}",
        dir=str(output_dir),
        config=asdict(cfg),
        reinit=True,
    )


def wandb_log(run, payload: Dict[str, object], step: Optional[int] = None) -> None:
    """Log metrics to Weights & Biases if run is active."""
    if run is not None:
        run.log(payload, step=step)


def print_run_header(
    model_name: str,
    setting_name: str,
    seed: int,
    device: torch.device,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: BenchmarkConfig,
) -> None:
    """Print training run header with configuration."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    multi_gpu = " (multi-GPU)" if gpu_count > 1 and cfg.train_on_gpu_if_available and cfg.use_multi_gpu else ""
    
    print("=" * 100, flush=True)
    print(
        f"[RunStart] model={model_name} setting={setting_name} seed={seed} device={device}{multi_gpu} "
        f"amp={cfg.use_amp and device.type == 'cuda'} batch_size={cfg.batch_size} workers={cfg.num_workers} lr={cfg.learning_rate}",
        flush=True,
    )
    if gpu_count > 1 and cfg.train_on_gpu_if_available and cfg.use_multi_gpu:
        print(f"[MultiGPU] Using {gpu_count} GPUs with DataParallel", flush=True)
    print(
        f"[Data] train={len(train_df)} ({format_label_distribution(train_df)}) | "
        f"val={len(val_df)} ({format_label_distribution(val_df)}) | "
        f"test={len(test_df)} ({format_label_distribution(test_df)})",
        flush=True,
    )


def train_one_setting(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    setting_name: str,
    seed: int,
    cfg: BenchmarkConfig,
    output_dir: Path,
) -> Dict[str, object]:
    """
    Train and evaluate a model for one benchmark setting.

    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        model_name: Model architecture name
        setting_name: Benchmark setting name
        seed: Random seed
        cfg: Benchmark configuration
        output_dir: Output directory for this run

    Returns:
        Dictionary with training results and metrics
    """
    set_seed(seed)
    device = torch.device(
        "cuda" if (cfg.train_on_gpu_if_available and torch.cuda.is_available()) else "cpu"
    )

    # Build data loaders
    train_tf, eval_tf = build_transforms(cfg.image_size)
    train_loader = make_train_loader(train_df, train_tf, cfg)
    val_loader = make_eval_loader(val_df, eval_tf, cfg)
    test_loader = make_eval_loader(test_df, eval_tf, cfg)

    print_run_header(
        model_name, setting_name, seed, device, train_df, val_df, test_df, cfg
    )
    run = maybe_init_wandb(cfg, model_name, setting_name, seed, output_dir)

    # Build model and training components
    ordinal_method = cfg.ordinal_methods[0] if cfg.ordinal_methods else None
    model = build_model(model_name, ordinal_method=ordinal_method)
    
    # Multi-GPU support using DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and cfg.train_on_gpu_if_available and cfg.use_multi_gpu:
        print(f"[MultiGPU] Using {torch.cuda.device_count()} GPUs with DataParallel", flush=True)
        model = nn.DataParallel(model)
    
    model = model.to(device)
    criterion = compute_loss_fn(train_df, cfg, device, ordinal_method=ordinal_method)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = make_grad_scaler(cfg.use_amp and device.type == "cuda")

    # Training loop
    best_val_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    train_rows = []
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_seen = 0

        for step, (x, y, _) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, cfg.use_amp):
                logits = model(x)
                # Compute loss based on ordinal method
                if ordinal_method == "coral":
                    loss = compute_coral_loss(logits, y)
                elif ordinal_method == "corn":
                    loss = compute_corn_loss(logits, y)
                else:
                    loss = criterion(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            n_seen += bs

            if step % cfg.log_interval == 0 or step == len(train_loader):
                print(
                    f"[Train] model={model_name} setting={setting_name} seed={seed} epoch={epoch}/{cfg.epochs} "
                    f"step={step}/{len(train_loader)} avg_loss={running_loss / max(n_seen, 1):.4f}",
                    flush=True,
                )

        train_loss = running_loss / max(n_seen, 1)
        val_metrics = evaluate_model(model, val_loader, device, ordinal_method=ordinal_method)
        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = val_metrics["macro_f1"] > best_val_macro_f1
        scheduler.step(val_metrics["macro_f1"])

        if improved:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        train_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_kappa": val_metrics["weighted_kappa"],
                "val_mae": val_metrics["mae"],
                "val_within1": val_metrics["within1"],
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
                "improved": int(improved),
            }
        )

        wandb_log(
            run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/accuracy": val_metrics["accuracy"],
                "val/macro_f1": val_metrics["macro_f1"],
                "val/weighted_kappa": val_metrics["weighted_kappa"],
                "val/mae": val_metrics["mae"],
                "val/within1": val_metrics["within1"],
                "train/lr": current_lr,
                "system/epoch_time_sec": epoch_time,
            },
            step=epoch,
        )

        print(
            f"[EpochEnd] model={model_name} setting={setting_name} seed={seed} epoch={epoch}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_kappa={val_metrics['weighted_kappa']:.4f} val_mae={val_metrics['mae']:.4f} lr={current_lr:.6g} time={epoch_time:.1f}s "
            f"{'BEST' if improved else f'patience={patience_counter}/{cfg.patience}'}",
            flush=True,
        )

        if patience_counter >= cfg.patience:
            print(
                f"[EarlyStop] model={model_name} setting={setting_name} seed={seed} stop_epoch={epoch} best_epoch={best_epoch} best_val_macro_f1={best_val_macro_f1:.4f}",
                flush=True,
            )
            break

    if best_state is None:
        if run is not None:
            run.finish()
        raise RuntimeError("Training did not produce a best checkpoint.")

    # Final evaluation
    model.load_state_dict(best_state)
    val_metrics = evaluate_model(model, val_loader, device, ordinal_method=ordinal_method)
    test_metrics = evaluate_model(model, test_loader, device, ordinal_method=ordinal_method)
    total_time = time.time() - start

    # Save outputs
    save_table(
        pd.DataFrame(train_rows),
        output_dir / f"train_log_{model_name}_{setting_name}_seed{seed}.csv",
    )
    if cfg.save_best_checkpoints:
        torch.save(
            best_state, output_dir / f"best_{model_name}_{setting_name}_seed{seed}.pt"
        )

    # Save predictions
    pred_rows = []
    from cleancam_pipeline.core.constants import INDEX_TO_LABEL

    for img_id, t_label, p_label, prob_vec in zip(
        test_metrics["image_ids"],
        test_metrics["true_labels"],
        test_metrics["pred_labels"],
        test_metrics["probs"],
    ):
        row = {
            "image_id": img_id,
            "target_label": int(t_label),
            "pred_label": int(p_label),
        }
        for idx, label in INDEX_TO_LABEL.items():
            row[f"prob_L{label}"] = float(prob_vec[idx])
        pred_rows.append(row)
    save_table(
        pd.DataFrame(pred_rows),
        output_dir / f"test_predictions_{model_name}_{setting_name}_seed{seed}.csv",
    )

    # Log final metrics to wandb
    wandb_payload = {
        "final/best_epoch": best_epoch,
        "final/total_time_sec": total_time,
        "test/accuracy": test_metrics["accuracy"],
        "test/macro_f1": test_metrics["macro_f1"],
        "test/weighted_kappa": test_metrics["weighted_kappa"],
        "test/mae": test_metrics["mae"],
        "test/within1": test_metrics["within1"],
        "test/binary_precision": test_metrics["binary_metrics"]["precision"],
        "test/binary_recall": test_metrics["binary_metrics"]["recall"],
        "test/binary_f1": test_metrics["binary_metrics"]["f1"],
        "test/binary_auroc": test_metrics["binary_metrics"]["auroc"],
        "test/binary_auprc": test_metrics["binary_metrics"]["auprc"],
    }
    for row in test_metrics["per_class_rows"]:
        label = row["label"]
        wandb_payload[f"test/L{label}_precision"] = row["precision"]
        wandb_payload[f"test/L{label}_recall"] = row["recall"]
        wandb_payload[f"test/L{label}_f1"] = row["f1"]

    wandb_log(run, wandb_payload, step=best_epoch if best_epoch > 0 else None)
    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_macro_f1"] = best_val_macro_f1
        run.finish()

    print(
        f"[Test] model={model_name} setting={setting_name} seed={seed} best_epoch={best_epoch} "
        f"test_acc={test_metrics['accuracy']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f} "
        f"test_kappa={test_metrics['weighted_kappa']:.4f} test_mae={test_metrics['mae']:.4f} total_time={total_time/60.0:.1f} min",
        flush=True,
    )

    return {
        "model_name": model_name,
        "setting_name": setting_name,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "total_time_sec": float(total_time),
    }

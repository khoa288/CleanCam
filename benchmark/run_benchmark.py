#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v2,
    resnet18,
)
from tqdm import tqdm

LABELS: Tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44)
DEFAULT_MODELS: Tuple[str, ...] = ("mobilenet_v2", "resnet18", "efficientnet_b0")
DEFAULT_TRAIN_SPLITS: Tuple[str, ...] = ("train_real_only", "train_real_plus_synthetic")
DEFAULT_EVAL_DOMAINS: Tuple[str, ...] = ("real_only", "real_plus_synthetic")


@dataclass
class BenchmarkConfig:
    dataset_root: str
    output_root: str
    models: Tuple[str, ...] = DEFAULT_MODELS
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    train_splits: Tuple[str, ...] = DEFAULT_TRAIN_SPLITS
    eval_domains: Tuple[str, ...] = DEFAULT_EVAL_DOMAINS
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    image_size: int = 224
    num_workers: int = min(8, os.cpu_count() or 4)
    weights: str = "none"
    optimizer: str = "adam"
    selection_metric: str = "qwk"
    amp: bool = True
    pin_memory: bool = True
    use_wandb: bool = False
    wandb_project: str = "cleancam-benchmark"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"


class SplitDataset(Dataset):
    def __init__(self, dataset_root: Path, split_csv: Path, transform: transforms.Compose):
        self.dataset_root = dataset_root
        self.transform = transform
        self.df = pd.read_csv(split_csv)
        required = {"relative_path", "label", "image_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {split_csv}: {sorted(missing)}")
        self.df = self.df.reset_index(drop=True)
        self.paths = [self.dataset_root / str(rel) for rel in self.df["relative_path"].tolist()]
        self.labels = [int(x) for x in self.df["label"].tolist()]
        self.image_ids = [str(x) for x in self.df["image_id"].tolist()]

        if self.paths:
            first_missing = next((p for p in self.paths if not p.exists()), None)
            if first_missing is not None:
                raise FileNotFoundError(f"Missing image file: {first_missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return {
            "image": image,
            "label": label - 1,
            "label_raw": label,
            "image_id": self.image_ids[idx],
        }


class EarlyStopping:
    def __init__(self, mode: str, patience: int):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be max or min")
        self.mode = mode
        self.patience = patience
        self.best_score: Optional[float] = None
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "max" and score > self.best_score:
            improved = True
        elif self.mode == "min" and score < self.best_score:
            improved = True

        if improved:
            self.best_score = score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs > self.patience


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run the CleanCam public benchmark on the official split CSVs")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--train-splits", nargs="+", default=list(DEFAULT_TRAIN_SPLITS), choices=list(DEFAULT_TRAIN_SPLITS))
    parser.add_argument("--eval-domains", nargs="+", default=list(DEFAULT_EVAL_DOMAINS), choices=list(DEFAULT_EVAL_DOMAINS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--weights", choices=["none", "imagenet"], default="none")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--selection-metric", choices=["qwk", "macro_f1", "loss"], default="qwk")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cleancam-benchmark")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()

    return BenchmarkConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        models=tuple(args.models),
        train_splits=tuple(args.train_splits),
        eval_domains=tuple(args.eval_domains),
        seeds=tuple(args.seeds),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        image_size=args.image_size,
        num_workers=args.num_workers,
        weights=args.weights,
        optimizer=args.optimizer,
        selection_metric=args.selection_metric,
        amp=not args.no_amp,
        pin_memory=not args.no_pin_memory,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transform(cfg: BenchmarkConfig) -> transforms.Compose:
    ops: List[object] = [transforms.Resize((cfg.image_size, cfg.image_size)), transforms.ToTensor()]
    if cfg.weights == "imagenet":
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


def split_paths(dataset_root: Path, train_split: str, eval_domain: str) -> Tuple[Path, Path, Path]:
    split_root = dataset_root / "splits" / "official"
    val_split = f"val_{eval_domain}"
    test_split = f"test_{eval_domain}"
    return split_root / f"{train_split}.csv", split_root / f"{val_split}.csv", split_root / f"{test_split}.csv"


def build_model(model_name: str, weights_mode: str) -> nn.Module:
    if model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if weights_mode == "imagenet" else None
        model = mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
        return model
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if weights_mode == "imagenet" else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        return model
    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if weights_mode == "imagenet" else None
        model = efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def build_optimizer(model: nn.Module, cfg: BenchmarkConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def make_train_sampler(dataset: SplitDataset) -> WeightedRandomSampler:
    labels = np.array(dataset.labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=max(LABELS) + 1)
    weights = np.zeros_like(labels, dtype=np.float64)
    for idx, label in enumerate(labels):
        weights[idx] = 1.0 / class_counts[label]
    generator = torch.Generator()
    generator.manual_seed(0)
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True, generator=generator)


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    labels_raw = torch.tensor([int(item["label_raw"]) for item in batch], dtype=torch.long)
    image_ids = [str(item["image_id"]) for item in batch]
    return {"image": images, "label": labels, "label_raw": labels_raw, "image_id": image_ids}


def create_loaders(cfg: BenchmarkConfig, dataset_root: Path, train_csv: Path, val_csv: Path, test_csv: Path):
    transform = build_transform(cfg)
    train_dataset = SplitDataset(dataset_root, train_csv, transform)
    val_dataset = SplitDataset(dataset_root, val_csv, transform)
    test_dataset = SplitDataset(dataset_root, test_csv, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=make_train_sampler(train_dataset),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_batch,
    )
    eval_loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
    return running_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, object]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_ids: List[str] = []
    running_loss = 0.0

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        labels_raw = batch["label_raw"].cpu().numpy().astype(int)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1) + 1
        running_loss += float(loss.item()) * images.size(0)

        all_labels.extend(labels_raw.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_ids.extend(batch["image_id"])

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    metrics["predictions"] = pd.DataFrame(
        {
            "image_id": all_ids,
            "label": all_labels,
            "prediction": all_preds,
            **{f"prob_label_{label}": np.array(all_probs)[:, label - 1] for label in LABELS},
        }
    )
    return metrics


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    multiclass = {
        "accuracy": float((labels == preds).mean()),
        "macro_f1": float(f1_score(labels, preds, labels=list(LABELS), average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "mae": float(np.mean(np.abs(labels - preds))),
        "within_1": float(np.mean(np.abs(labels - preds) <= 1)),
    }

    y_true_bin = (labels >= 4).astype(int)
    y_pred_bin = (preds >= 4).astype(int)
    y_score_bin = probs[:, 3] + probs[:, 4]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="binary", zero_division=0)
    binary = {
        "binary_precision": float(precision),
        "binary_recall": float(recall),
        "binary_f1": float(f1),
        "binary_auroc": float(roc_auc_score(y_true_bin, y_score_bin)) if len(np.unique(y_true_bin)) > 1 else float("nan"),
        "binary_auprc": float(average_precision_score(y_true_bin, y_score_bin)) if len(np.unique(y_true_bin)) > 1 else float("nan"),
    }
    return {**multiclass, **binary}


def selection_score(metrics: Dict[str, float], selection_metric: str) -> float:
    if selection_metric == "loss":
        return float(metrics["loss"])
    if selection_metric == "macro_f1":
        return float(metrics["macro_f1"])
    return float(metrics["qwk"])


def selection_mode(selection_metric: str) -> str:
    return "min" if selection_metric == "loss" else "max"


def maybe_init_wandb(cfg: BenchmarkConfig, run_name: str, run_dir: Path, extra_config: Dict[str, object]):
    if not cfg.use_wandb or cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        dir=str(run_dir),
        mode=cfg.wandb_mode,
        config={**asdict(cfg), **extra_config},
    )


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def aggregate_runs(df: pd.DataFrame, metric_names: Sequence[str], group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group_values, subdf in df.groupby(list(group_cols), sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {col: value for col, value in zip(group_cols, group_values)}
        row["n_runs"] = int(len(subdf))
        for metric in metric_names:
            values = [float(v) for v in subdf[metric].tolist()]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_single(cfg: BenchmarkConfig, model_name: str, train_split: str, eval_domain: str, seed: int) -> Dict[str, object]:
    seed_everything(seed)
    dataset_root = Path(cfg.dataset_root).resolve()
    output_root = Path(cfg.output_root).resolve()
    run_dir = output_root / f"train-{train_split}" / f"eval-{eval_domain}" / model_name / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_csv, val_csv, test_csv = split_paths(dataset_root, train_split, eval_domain)
    train_loader, val_loader, test_loader = create_loaders(cfg, dataset_root, train_csv, val_csv, test_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, cfg.weights).to(device)
    optimizer = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if cfg.amp and device.type == "cuda" else None
    stopper = EarlyStopping(mode=selection_mode(cfg.selection_metric), patience=cfg.patience)

    best_checkpoint = run_dir / "best.pt"
    history: List[Dict[str, float]] = []
    wb_run = maybe_init_wandb(
        cfg,
        run_name=f"{model_name}-{train_split}-{eval_domain}-seed{seed}",
        run_dir=run_dir,
        extra_config={"model": model_name, "train_split": train_split, "eval_domain": eval_domain, "seed": seed},
    )

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg.amp)
        val_metrics = evaluate(model, val_loader, criterion, device, cfg.amp)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_qwk": float(val_metrics["qwk"]),
            "val_mae": float(val_metrics["mae"]),
            "val_within_1": float(val_metrics["within_1"]),
        }
        history.append(row)
        if wb_run is not None:
            wb_run.log(row)

        score = selection_score(val_metrics, cfg.selection_metric)
        improved = False
        if stopper.best_score is None:
            improved = True
        elif stopper.mode == "max" and score > stopper.best_score:
            improved = True
        elif stopper.mode == "min" and score < stopper.best_score:
            improved = True
        if improved:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, best_checkpoint)

        if stopper.step(score):
            break

    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_metrics = evaluate(model, val_loader, criterion, device, cfg.amp)
    test_metrics = evaluate(model, test_loader, criterion, device, cfg.amp)

    val_pred = val_metrics.pop("predictions")
    test_pred = test_metrics.pop("predictions")
    val_pred.to_csv(run_dir / "val_predictions.csv", index=False)
    test_pred.to_csv(run_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    conf = confusion_matrix(test_pred["label"].to_numpy(), test_pred["prediction"].to_numpy(), labels=list(LABELS), normalize="true")
    pd.DataFrame(conf, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS]).to_csv(run_dir / "test_confusion_matrix_normalized.csv")

    summary = {
        "model": model_name,
        "train_split": train_split,
        "eval_domain": eval_domain,
        "seed": seed,
        **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        **{f"test_{k}": float(v) for k, v in test_metrics.items()},
    }
    save_json(summary, run_dir / "metrics.json")

    if wb_run is not None:
        wb_run.summary.update(summary)
        wb_run.finish()

    return summary


def write_summaries(cfg: BenchmarkConfig, all_runs: pd.DataFrame) -> None:
    output_root = Path(cfg.output_root).resolve()
    summary_root = output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    all_runs.to_csv(summary_root / "all_runs.csv", index=False)

    multiclass_metrics = ["test_accuracy", "test_macro_f1", "test_qwk", "test_mae", "test_within_1"]
    binary_metrics = ["test_binary_precision", "test_binary_recall", "test_binary_f1", "test_binary_auroc", "test_binary_auprc"]

    aggregate_all = aggregate_runs(all_runs, multiclass_metrics + binary_metrics, ["train_split", "eval_domain", "model"])
    aggregate_all.to_csv(summary_root / "aggregate_all_settings.csv", index=False)

    primary = all_runs.loc[all_runs["eval_domain"] == "real_only"].copy()
    primary_5class = aggregate_runs(primary, multiclass_metrics, ["train_split", "model"])
    primary_binary = aggregate_runs(primary, binary_metrics, ["train_split", "model"])
    primary_5class.to_csv(summary_root / "paper_table_5class_real_only.csv", index=False)
    primary_binary.to_csv(summary_root / "paper_table_binary_real_only.csv", index=False)


def main() -> None:
    cfg = parse_args()
    results: List[Dict[str, object]] = []
    for train_split in cfg.train_splits:
        for eval_domain in cfg.eval_domains:
            for model_name in cfg.models:
                for seed in cfg.seeds:
                    print(f"Running model={model_name} train_split={train_split} eval_domain={eval_domain} seed={seed}")
                    results.append(run_single(cfg, model_name, train_split, eval_domain, seed))

    all_runs = pd.DataFrame(results)
    write_summaries(cfg, all_runs)
    print(f"Done. Results written to: {Path(cfg.output_root).resolve()}")


if __name__ == "__main__":
    main()

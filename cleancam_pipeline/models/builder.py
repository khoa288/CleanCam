"""Model architecture builders."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models

from cleancam_pipeline.core.config import BenchmarkConfig
from cleancam_pipeline.core.constants import LABELS
from cleancam_pipeline.models.ordinal import CoralHead, CornHead


def build_model(
    model_name: str, num_classes: int = 5, ordinal_method: str = None
) -> nn.Module:
    """
    Build a CNN model with pretrained weights.

    Args:
        model_name: Name of model architecture
        num_classes: Number of output classes
        ordinal_method: Ordinal regression method ('coral', 'corn', or None for standard)

    Returns:
        PyTorch model

    Raises:
        ValueError: If model_name is not supported
    """
    # Build base model
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        
        if ordinal_method == "coral":
            model.fc = CoralHead(in_features, num_classes)
        elif ordinal_method == "corn":
            model.fc = CornHead(in_features, num_classes)
        else:
            model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        
        if ordinal_method == "coral":
            model.classifier[1] = CoralHead(in_features, num_classes)
        elif ordinal_method == "corn":
            model.classifier[1] = CornHead(in_features, num_classes)
        else:
            model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        
        if ordinal_method == "coral":
            model.classifier[1] = CoralHead(in_features, num_classes)
        elif ordinal_method == "corn":
            model.classifier[1] = CornHead(in_features, num_classes)
        else:
            model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def compute_loss_fn(
    train_df: pd.DataFrame,
    cfg: BenchmarkConfig,
    device: torch.device,
    ordinal_method: str = None,
) -> nn.Module:
    """
    Create loss function with optional class weights.

    Args:
        train_df: Training data DataFrame
        cfg: Benchmark configuration
        device: Target device for weights
        ordinal_method: Ordinal regression method ('coral', 'corn', or None)

    Returns:
        Loss function module or callable
    """
    # For ordinal methods, return None (loss computed in training loop)
    if ordinal_method in ["coral", "corn"]:
        return None
    
    # Standard cross-entropy loss
    if cfg.use_class_weights:
        y = train_df["label"].astype(int).to_numpy()
        weights = compute_class_weight(
            class_weight="balanced", classes=np.array(LABELS), y=y
        )
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)

    return nn.CrossEntropyLoss()

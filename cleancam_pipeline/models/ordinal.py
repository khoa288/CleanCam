"""Ordinal regression models and losses (CORAL & CORN)."""

from typing import Tuple

import torch
import torch.nn as nn
from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss


class OrdinalHead(nn.Module):
    """Base class for ordinal regression heads."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes


class CoralHead(OrdinalHead):
    """
    CORAL (Consistent Rank Logits) ordinal regression head.
    
    Reference: https://arxiv.org/abs/1901.07884
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__(in_features, num_classes)
        self.coral_layer = CoralLayer(size_in=in_features, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for binary tasks."""
        return self.coral_layer(x)


class CornHead(OrdinalHead):
    """
    CORN (Conditional Ordinal Regression for Neural networks) head.
    
    CORN uses a simple linear layer with num_classes-1 outputs.
    
    Reference: https://arxiv.org/abs/2111.08851
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__(in_features, num_classes)
        # CORN uses a regular linear layer with num_classes-1 outputs
        self.linear = nn.Linear(in_features, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for conditional tasks."""
        return self.linear(x)


def compute_coral_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute CORAL loss.
    
    Args:
        logits: Model output logits [batch_size, num_classes-1]
        labels: True labels [batch_size] with values in [0, num_classes-1]
    
    Returns:
        CORAL loss value
    """
    # Convert labels to levels (binary encoding)
    levels = levels_from_labelbatch(labels, num_classes=logits.size(1) + 1)
    return coral_loss(logits, levels)


def compute_corn_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute CORN loss.
    
    Args:
        logits: Model output logits [batch_size, num_classes-1]
        labels: True labels [batch_size] with values in [0, num_classes-1]
    
    Returns:
        CORN loss value
    """
    return corn_loss(logits, labels, num_classes=logits.size(1) + 1)


def predict_from_coral_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL logits to predicted labels.
    
    Args:
        logits: CORAL logits [batch_size, num_classes-1]
    
    Returns:
        Predicted labels [batch_size]
    """
    # Sum of probabilities gives the predicted rank
    probas = torch.sigmoid(logits)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


def predict_from_corn_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN logits to predicted labels.
    
    Args:
        logits: CORN logits [batch_size, num_classes-1]
    
    Returns:
        Predicted labels [batch_size]
    """
    return corn_label_from_logits(logits)


def get_ordinal_probabilities(
    logits: torch.Tensor, method: str
) -> torch.Tensor:
    """
    Convert ordinal logits to class probabilities.
    
    Args:
        logits: Ordinal logits [batch_size, num_classes-1]
        method: 'coral' or 'corn'
    
    Returns:
        Class probabilities [batch_size, num_classes]
    """
    if method == "coral":
        # CORAL: Convert cumulative probabilities to class probabilities
        cumulative_probs = torch.sigmoid(logits)
        # Add boundaries
        cumulative_probs = torch.cat(
            [
                torch.zeros(cumulative_probs.size(0), 1, device=logits.device),
                cumulative_probs,
                torch.ones(cumulative_probs.size(0), 1, device=logits.device),
            ],
            dim=1,
        )
        # Compute class probabilities
        class_probs = cumulative_probs[:, 1:] - cumulative_probs[:, :-1]
        return class_probs
    
    elif method == "corn":
        # CORN: Convert conditional probabilities to class probabilities
        conditional_probs = torch.sigmoid(logits)
        # Compute class probabilities using the chain rule
        class_probs = []
        for k in range(logits.size(1) + 1):
            if k == 0:
                # P(Y=0) = 1 - P(Y>0)
                prob = 1.0 - conditional_probs[:, 0]
            elif k == logits.size(1):
                # P(Y=K) = P(Y>K-1)
                prob = conditional_probs[:, k - 1]
            else:
                # P(Y=k) = P(Y>k-1) - P(Y>k)
                prob = conditional_probs[:, k - 1] - conditional_probs[:, k]
            class_probs.append(prob.unsqueeze(1))
        return torch.cat(class_probs, dim=1)
    
    else:
        raise ValueError(f"Unknown ordinal method: {method}")

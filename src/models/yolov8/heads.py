"""
Detection heads for YOLOv8.

Provides modular, reusable detection head components.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DetectionHead(nn.Module):
    """Base class for detection heads."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 256):
        """
        Initialize detection head.

        Args:
            in_channels: Number of input channels from backbone
            out_channels: Number of output channels
            hidden_channels: Number of hidden channels
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.head(x)


class BoxRegressionHead(DetectionHead):
    """
    Box regression head.

    Predicts 4 values per anchor: (tx, ty, tw, th)
    These are decoded to (x1, y1, x2, y2) using grid and stride.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__(in_channels, out_channels=4, hidden_channels=hidden_channels)


class ClassificationHead(DetectionHead):
    """
    Classification head.

    Predicts class logits for each anchor.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 256, dropout: float = 0.3):
        super().__init__(in_channels, out_channels=num_classes, hidden_channels=hidden_channels)
        # Add dropout before final layer for regularization
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        )


class ObjectnessHead(DetectionHead):
    """
    Objectness head.

    Predicts a single objectness score per anchor indicating
    whether the anchor contains an object vs background.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__(in_channels, out_channels=1, hidden_channels=hidden_channels)


class YOLOv8Head(nn.Module):
    """
    Complete YOLOv8 detection head.

    Combines box regression, classification, and objectness predictions.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        box_hidden: int = 256,
        cls_hidden: int = 256,
        obj_hidden: int = 128
    ):
        """
        Initialize YOLOv8 head.

        Args:
            in_channels: Number of input channels from backbone
            num_classes: Number of object classes
            dropout: Dropout rate for classification head
            box_hidden: Hidden channels for box head
            cls_hidden: Hidden channels for classification head
            obj_hidden: Hidden channels for objectness head
        """
        super().__init__()
        self.num_classes = num_classes

        self.box_head = BoxRegressionHead(
            in_channels, hidden_channels=box_hidden)
        self.cls_head = ClassificationHead(
            in_channels, num_classes, hidden_channels=cls_hidden, dropout=dropout)
        self.obj_head = ObjectnessHead(in_channels, hidden_channels=obj_hidden)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Feature map from backbone [B, C, H, W]

        Returns:
            Tuple of (box_preds, cls_preds, obj_preds)
            - box_preds: [B, 4, H, W] box parameters
            - cls_preds: [B, num_classes, H, W] class logits
            - obj_preds: [B, 1, H, W] objectness logits
        """
        box_preds = self.box_head(features)  # [B, 4, H, W]
        cls_preds = self.cls_head(features)  # [B, C, H, W]
        obj_preds = self.obj_head(features)  # [B, 1, H, W]

        return box_preds, cls_preds, obj_preds

"""
Abstract base class for all detection models.

Provides a consistent interface for training, inference, and loss computation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn


class DetectionModelBase(nn.Module, ABC):
    """
    Abstract base class for object detection models.

    All detection models should inherit from this class and implement
    the required methods.
    """

    def __init__(self, num_classes: int, **kwargs):
        """
        Initialize the detection model.

        Args:
            num_classes: Number of object classes (excluding background)
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.num_classes = num_classes
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Any:
        """
        Forward pass of the model.

        Args:
            images: Batch of images [B, C, H, W]
            targets: Ground truth targets for training (optional)
                Each target dict should contain:
                - 'boxes': [N, 4] in (x1, y1, x2, y2) format
                - 'labels': [N] class labels

        Returns:
            Training mode: Model-specific output for loss computation
            Inference mode: List of detection dicts, one per image
                Each dict contains:
                - 'boxes': [M, 4] predicted boxes
                - 'scores': [M] confidence scores
                - 'labels': [M] predicted class labels
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        outputs: Any,
        targets: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute the training loss.

        Args:
            outputs: Model outputs from forward pass
            targets: Ground truth targets

        Returns:
            Loss tensor with gradient enabled
        """
        pass

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information dictionary."""
        return {
            'name': self.model_name,
            'num_classes': self.num_classes,
            'total_params': self.get_num_params(),
            'trainable_params': self.get_num_trainable_params(),
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"{self.model_name}(\n"
            f"  num_classes={info['num_classes']},\n"
            f"  total_params={info['total_params']:,},\n"
            f"  trainable_params={info['trainable_params']:,}\n"
            f")"
        )

"""
YOLOv8 Model Architecture.

Clean, modular implementation of YOLOv8 detection model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

from ...core.model_base import DetectionModelBase
from ...core.registry import ModelRegistry
from .heads import YOLOv8Head
from .config import get_default_config, update_config
from .loss import YOLOv8Loss
from ...utils.box_utils import box_cxcywh_to_xyxy

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = 'silu'
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross Stage Partial Block for efficient feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3):
        super().__init__()
        hidden_channels = out_channels // 2

        # Split convolution
        self.conv1 = ConvBlock(
            in_channels, hidden_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(
            in_channels, hidden_channels, kernel_size=1, padding=0)

        # Bottleneck blocks
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvBlock(hidden_channels, hidden_channels,
                          kernel_size=1, padding=0),
                ConvBlock(hidden_channels, hidden_channels,
                          kernel_size=3, padding=1)
            )
            for _ in range(num_blocks)
        ])

        # Fusion convolution
        self.conv3 = ConvBlock(hidden_channels * 2,
                               out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split path
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        # Process through blocks
        x2 = self.blocks(x2)

        # Concatenate and fuse
        out = torch.cat([x1, x2], dim=1)
        out = self.conv3(out)

        return out


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast version."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2

        self.conv1 = ConvBlock(
            in_channels, hidden_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(hidden_channels * 4,
                               out_channels, kernel_size=1, padding=0)
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        # Multi-scale pooling
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)

        # Concatenate all scales
        out = torch.cat([x, y1, y2, y3], dim=1)
        out = self.conv2(out)

        return out


class YOLOv8Backbone(nn.Module):
    """YOLOv8 Backbone Network."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # Stem
        self.stem = ConvBlock(in_channels, base_channels,
                              kernel_size=3, stride=2, padding=1)

        # Stage 1: 64 -> 128
        self.stage1 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            CSPBlock(base_channels * 2, base_channels * 2, num_blocks=3)
        )

        # Stage 2: 128 -> 256
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4,
                      kernel_size=3, stride=2, padding=1),
            CSPBlock(base_channels * 4, base_channels * 4, num_blocks=6)
        )

        # Stage 3: 256 -> 512
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8,
                      kernel_size=3, stride=2, padding=1),
            CSPBlock(base_channels * 8, base_channels * 8, num_blocks=6)
        )

        # Stage 4: 512 -> 512 with SPPF
        self.stage4 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 8,
                      kernel_size=3, stride=2, padding=1),
            CSPBlock(base_channels * 8, base_channels * 8, num_blocks=3),
            SPPF(base_channels * 8, base_channels * 8)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through backbone.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            List of feature maps at different scales
        """
        x = self.stem(x)

        c1 = self.stage1(x)   # /4
        c2 = self.stage2(c1)  # /8
        c3 = self.stage3(c2)  # /16
        c4 = self.stage4(c3)  # /32

        return [c2, c3, c4]  # Return multi-scale features


@ModelRegistry.register('yolov8')
class YOLOv8Model(DetectionModelBase):
    """
    YOLOv8 Detection Model.

    A modular, clean implementation following SOLID principles.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 64,
        config: Optional[Dict] = None
    ):
        """
        Initialize YOLOv8 model.

        Args:
            num_classes: Number of object classes
            in_channels: Number of input image channels
            base_channels: Base number of channels in backbone
            config: Optional configuration dict (overrides defaults)
        """
        super().__init__(num_classes=num_classes)

        # Configuration
        self.config = get_default_config()
        if config is not None:
            self.config = update_config(self.config, config)

        # Backbone
        self.backbone = YOLOv8Backbone(
            in_channels=in_channels,
            base_channels=base_channels
        )

        # Feature channels from backbone
        feat_channels = base_channels * 8  # 512 for default

        # Detection heads for multi-scale features
        self.heads = nn.ModuleList([
            YOLOv8Head(
                in_channels=feat_channels,
                num_classes=num_classes,
                dropout=self.config['dropout']
            )
            for _ in range(3)  # Three scales
        ])

        # Loss function
        self.loss_fn = YOLOv8Loss(
            num_classes=num_classes,
            box_weight=self.config['box_weight'],
            cls_weight=self.config['cls_weight'],
            obj_weight=self.config['obj_weight'],
            focal_alpha=self.config['focal_alpha'],
            focal_gamma=self.config['focal_gamma'],
            iou_thresh_start=self.config['iou_thresh'],
            iou_thresh_end=self.config['iou_thresh'],
            iou_warmup_epochs=30
        )

        # Anchor-free: direct prediction
        self.num_proposals = 400  # Total proposals from all scales

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through model.

        Args:
            images: Input images [B, 3, H, W]
            targets: Optional ground truth (used during training)

        Returns:
            Tuple of (box_preds, cls_preds, obj_preds)
            - box_preds: [B, N, 4] predicted boxes in xyxy format
            - cls_preds: [B, N, C] predicted class logits
            - obj_preds: [B, N, 1] predicted objectness logits
        """
        batch_size = images.size(0)
        device = images.device

        # Extract multi-scale features
        features = self.backbone(images)  # List of 3 feature maps

        # Collect predictions from all scales
        all_box_preds = []
        all_cls_preds = []
        all_obj_preds = []

        for i, (feat, head) in enumerate(zip(features, self.heads)):
            # Get predictions from head
            box_pred, cls_pred, obj_pred = head(feat)

            # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
            b, c, h, w = box_pred.shape
            box_pred = box_pred.permute(0, 2, 3, 1).reshape(b, -1, 4)

            b, c, h, w = cls_pred.shape
            cls_pred = cls_pred.permute(
                0, 2, 3, 1).reshape(b, -1, self.num_classes)

            b, c, h, w = obj_pred.shape
            obj_pred = obj_pred.permute(0, 2, 3, 1).reshape(b, -1, 1)

            # Convert boxes from cxcywh to xyxy format
            box_pred = box_cxcywh_to_xyxy(box_pred)

            # Collect
            all_box_preds.append(box_pred)
            all_cls_preds.append(cls_pred)
            all_obj_preds.append(obj_pred)

        # Concatenate predictions from all scales
        box_preds = torch.cat(all_box_preds, dim=1)  # [B, N, 4]
        cls_preds = torch.cat(all_cls_preds, dim=1)  # [B, N, C]
        obj_preds = torch.cat(all_obj_preds, dim=1)  # [B, N, 1]

        # Limit to num_proposals
        if box_preds.size(1) > self.num_proposals:
            box_preds = box_preds[:, :self.num_proposals]
            cls_preds = cls_preds[:, :self.num_proposals]
            obj_preds = obj_preds[:, :self.num_proposals]

        return box_preds, cls_preds, obj_preds

    def compute_loss(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute detection loss.

        Args:
            predictions: Tuple of (box_preds, cls_preds, obj_preds)
            targets: List of ground truth dicts

        Returns:
            Total loss value
        """
        return self.loss_fn(predictions, targets)

    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze/unfreeze backbone weights.

        Args:
            freeze: Whether to freeze (True) or unfreeze (False)
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        logger.info(f"Backbone {'frozen' if freeze else 'unfrozen'}")

    def get_model_info(self) -> Dict:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        return {
            'name': 'YOLOv8',
            'num_classes': self.num_classes,
            'num_proposals': self.num_proposals,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config
        }

"""
Faster R-CNN Model Implementation.

Modular Faster R-CNN with flexible backbone options.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from typing import Optional, Dict
import logging

from .config import get_default_config

logger = logging.getLogger(__name__)


class LightBackbone(nn.Module):
    """Lightweight backbone for faster training."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.out_channels = 64  # Required by FasterRCNN

    def forward(self, x):
        return self.features(x)


class FasterRCNNModel(nn.Module):
    """
    Faster R-CNN Detection Model.

    Modular implementation with configurable parameters.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_type: str = 'light',
        pretrained: bool = False,
        config: Optional[Dict] = None
    ):
        """
        Initialize Faster R-CNN model.

        Args:
            num_classes: Number of object classes (including background)
            backbone_type: Backbone architecture ('light', 'resnet50')
            pretrained: Use pretrained weights
            config: Optional configuration dict
        """
        super().__init__()

        self.num_classes = num_classes
        self.backbone_type = backbone_type

        # Load configuration
        self.config = get_default_config()
        if config is not None:
            self.config.update(config)

        # Build model
        self.model = self._build_model()

        logger.info(f"✅ Faster R-CNN with {backbone_type} backbone")
        logger.info(f"   Classes: {num_classes}")

    def _build_model(self) -> FasterRCNN:
        """Build Faster R-CNN model."""

        # Create backbone
        if self.backbone_type == 'light':
            backbone = LightBackbone()
            out_channels = 64
        elif self.backbone_type == 'resnet50':
            # Use torchvision ResNet50 backbone
            resnet = torchvision.models.resnet50(pretrained=True)
            backbone = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            )
            backbone.out_channels = 2048
            out_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_type}")

        # Anchor generator (one feature map → one tuple)
        anchor_generator = AnchorGenerator(
            sizes=self.config['anchor_sizes'],
            aspect_ratios=self.config['anchor_aspect_ratios']
        )

        # ROI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=self.config['roi_output_size'],
            sampling_ratio=self.config['roi_sampling_ratio']
        )

        # Build Faster R-CNN
        model = FasterRCNN(
            backbone=backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=self.config['min_size'],
            max_size=self.config['max_size'],
            rpn_pre_nms_top_n_train=self.config['rpn_pre_nms_top_n_train'],
            rpn_pre_nms_top_n_test=self.config['rpn_pre_nms_top_n_test'],
            rpn_post_nms_top_n_train=self.config['rpn_post_nms_top_n_train'],
            rpn_post_nms_top_n_test=self.config['rpn_post_nms_top_n_test'],
            rpn_batch_size_per_image=self.config['rpn_batch_size_per_image'],
            rpn_fg_iou_thresh=self.config['rpn_fg_iou_thresh'],
            rpn_bg_iou_thresh=self.config['rpn_bg_iou_thresh'],
            box_detections_per_img=self.config['box_detections_per_img'],
            box_score_thresh=self.config['box_score_thresh'],
            box_nms_thresh=self.config['box_nms_thresh'],
        )

        return model

    def forward(self, images, targets=None):
        """
        Forward pass.

        Args:
            images: Batch of images [B, C, H, W]
            targets: List of target dicts (training only)

        Returns:
            Training: dict of losses
            Inference: list of detection dicts
        """
        return self.model(images, targets)

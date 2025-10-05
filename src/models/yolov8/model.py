"""
YOLOv8 Model Architecture.

Clean, modular implementation of YOLOv8 detection model with flexible backbone options.
Supports: CSP (original YOLOv8), ResNet (18/34/50/101), ResNeXt (50/101)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import logging

from .heads import YOLOv8Head
from .config import get_default_config, update_config
from .loss import YOLOv8Loss
from .backbones import build_backbone

logger = logging.getLogger(__name__)


class YOLOv8Model(nn.Module):
    """
    YOLOv8 Detection Model with Flexible Backbone Options.

    Supports multiple backbones:
    - CSP: Original YOLOv8 backbone
    - ResNet: ResNet18/34/50/101 with pretrained weights
    - ResNeXt: ResNeXt50/101 for better performance
    """

    def __init__(
        self,
        num_classes: int,
        backbone_type: str = 'resnet50',
        pretrained: bool = True,
        in_channels: int = 3,
        base_channels: int = 64,
        config: Optional[Dict] = None
    ):
        """
        Initialize YOLOv8 model.

        Args:
            num_classes: Number of object classes
            backbone_type: Backbone architecture ('csp', 'resnet50', 'resnet101', etc.)
            pretrained: Use pretrained weights for ResNet/ResNeXt backbones
            in_channels: Number of input image channels
            base_channels: Base number of channels in CSP backbone
            config: Optional configuration dict (overrides defaults)
        """
        super().__init__()

        self.num_classes = num_classes
        self.backbone_type = backbone_type

        # Configuration
        self.config = get_default_config()
        if config is not None:
            self.config = update_config(self.config, config)

        # Build backbone (CSP, ResNet, or ResNeXt)
        self.backbone = build_backbone(
            backbone_type=backbone_type,
            pretrained=pretrained,
            in_channels=in_channels,
            base_channels=base_channels
        )

        # Feature channels from backbone (depends on backbone type)
        feat_channels = self.backbone.out_channels  # [c2, c3, c4] channels

        logger.info(f"✅ YOLOv8 with {self.backbone.name} backbone")
        logger.info(f"   Feature channels: {feat_channels}")

        # Detection heads for multi-scale features (one per scale)
        self.heads = nn.ModuleList([
            YOLOv8Head(
                in_channels=feat_channels[i],  # Different channels per scale
                num_classes=num_classes,
                dropout=self.config['dropout']
            )
            for i in range(3)  # Three scales: /8, /16, /32
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

        # Indicate expected input type for training loops
        self.input_type = 'tensor'

        # Anchor-free: direct prediction
        self.max_inference_proposals = self.config.get('max_proposals')

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
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through model.

        Args:
            images: Input images [B, 3, H, W]
            targets: Optional ground truth (used during training)

        Returns:
            Training: Dictionary of loss components
            Inference: Tuple of (box_preds, cls_preds, obj_preds)
                - box_preds: [B, N, 4] predicted boxes in xyxy format
                - cls_preds: [B, N, C] predicted class logits
                - obj_preds: [B, N, 1] predicted objectness logits
        """
        if isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("Received empty image batch")
            images = torch.stack(images, dim=0)

        batch_size = images.size(0)
        device = images.device

        # Extract multi-scale features
        features = self.backbone(images)  # List of 3 feature maps

        # Collect predictions from all scales
        all_box_preds = []
        all_cls_preds = []
        all_obj_preds = []

        input_height = images.size(2)
        input_width = images.size(3)

        for feat, head in zip(features, self.heads):
            # Get predictions from head
            box_pred, cls_pred, obj_pred = head(feat)

            b, _, h, w = box_pred.shape

            stride_y = input_height / h
            stride_x = input_width / w

            # Decode raw box predictions to absolute xyxy coordinates
            decoded_boxes = self._decode_boxes(
                box_pred,
                stride_x=stride_x,
                stride_y=stride_y,
                device=images.device,
            )  # [B, H*W, 4]

            # Flatten class/objectness logits
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
            obj_pred = obj_pred.permute(0, 2, 3, 1).reshape(b, -1, 1)

            all_box_preds.append(decoded_boxes)
            all_cls_preds.append(cls_pred)
            all_obj_preds.append(obj_pred)

        # Concatenate predictions from all scales
        box_preds = torch.cat(all_box_preds, dim=1)  # [B, N, 4]
        cls_preds = torch.cat(all_cls_preds, dim=1)  # [B, N, C]
        obj_preds = torch.cat(all_obj_preds, dim=1)  # [B, N, 1]

        if (not self.training) and self.max_inference_proposals:
            max_props = int(self.max_inference_proposals)
            if box_preds.size(1) > max_props:
                obj_logits = obj_preds.squeeze(-1)
                _, topk_idx = torch.topk(obj_logits, k=max_props, dim=1)

                idx_boxes = topk_idx.unsqueeze(-1).expand(-1, -1, 4)
                idx_cls = topk_idx.unsqueeze(-1).expand(-1, -1, self.num_classes)
                idx_obj = topk_idx.unsqueeze(-1)

                box_preds = torch.gather(box_preds, 1, idx_boxes)
                cls_preds = torch.gather(cls_preds, 1, idx_cls)
                obj_preds = torch.gather(obj_preds, 1, idx_obj)

        predictions = (box_preds, cls_preds, obj_preds)

        # During training, return loss components for the trainer
        if self.training and targets is not None:
            loss_dict = self.loss_fn(predictions, targets)
            return loss_dict

        return predictions

    def _decode_boxes(
        self,
        raw_boxes: torch.Tensor,
        stride_x: float,
        stride_y: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert raw head outputs into absolute xyxy boxes."""

        b, _, h, w = raw_boxes.shape

        # Prepare spatial grids (reuse on device for efficiency)
        if not hasattr(self, '_grid_cache'):
            self._grid_cache = {}

        grid_key = (h, w, raw_boxes.dtype, device)
        if grid_key not in self._grid_cache:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device, dtype=raw_boxes.dtype),
                torch.arange(w, device=device, dtype=raw_boxes.dtype),
                indexing='ij'
            )
            grid = torch.stack((xx, yy), dim=-1)  # [H, W, 2]
            self._grid_cache[grid_key] = grid
        else:
            grid = self._grid_cache[grid_key]

        raw_boxes = raw_boxes.permute(0, 2, 3, 1)  # [B, H, W, 4]
        tx = raw_boxes[..., 0]
        ty = raw_boxes[..., 1]
        tw = raw_boxes[..., 2].clamp(min=-5.0, max=5.0)
        th = raw_boxes[..., 3].clamp(min=-5.0, max=5.0)

        # Decode center coordinates relative to grid cell
        cx = (torch.sigmoid(tx) + grid[..., 0]) * stride_x
        cy = (torch.sigmoid(ty) + grid[..., 1]) * stride_y

        # Decode width/height (ensure positive values)
        width = torch.exp(tw) * stride_x
        height = torch.exp(th) * stride_y

        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2

        boxes = torch.stack((x1, y1, x2, y2), dim=-1)  # [B, H, W, 4]
        return boxes.reshape(b, -1, 4)

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
            'max_inference_proposals': self.max_inference_proposals,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config
        }

"""
Simplified detection model for cattle detection.
Uses ResNet50 backbone with single-scale detection head.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from typing import Dict, List

class SimplifiedDetector(nn.Module):
    """
    Simplified detection model with single-scale features.
    Uses ResNet50 backbone pretrained on ImageNet.
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # Background + cattle
        min_size: int = 800,
        max_size: int = 1333,
        rpn_pre_nms_top_n_train: int = 4000,  # Increased for better recall
        rpn_pre_nms_top_n_test: int = 2000,   # Increased for better recall
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.8,          # Increased to keep more overlapping proposals
        rpn_fg_iou_thresh: float = 0.6,       # Adjusted for better positive sample selection
        rpn_bg_iou_thresh: float = 0.3,
        box_score_thresh: float = 0.3,        # Increased for higher confidence detections
        box_nms_thresh: float = 0.45,         # Adjusted to reduce duplicate detections
        box_detections_per_img: int = 50      # Reduced as we expect ~8 cattle per image
    ):
        super().__init__()
        
        # Load pretrained ResNet50
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the last two layers (avgpool and fc)
        layers = list(backbone.children())[:-2]
        
        # Create new backbone with required output channels
        self.backbone = nn.Sequential(*layers)
        
        # Set output channels (2048 for ResNet50's last layer)
        self.backbone.out_channels = 2048
        
        # Create anchor generator optimized for cattle dataset
        # Based on detailed box distribution analysis
        anchor_generator = AnchorGenerator(
            sizes=((192, 384, 768),),  # Optimized for typical cattle sizes in dataset
            aspect_ratios=((0.7, 1.0, 1.3),)  # Cattle-specific aspect ratios
        )
        
        # Create ROI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Create the detector
        self.detector = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=min_size,
            max_size=max_size,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_fg_iou_thresh=rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=rpn_bg_iou_thresh,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize detection head weights."""
        for name, param in self.detector.named_parameters():
            if "box_predictor" in name:
                if "weight" in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)
                    
    def forward(self, 
               images: List[torch.Tensor], 
               targets: List[Dict[str, torch.Tensor]] = None):
        """
        Forward pass with automatic training/inference mode.
        
        Args:
            images: List of images
            targets: List of target dictionaries with 'boxes' and 'labels'
            
        Returns:
            During training: Loss dict
            During inference: List of predictions
        """
        return self.detector(images, targets)

def create_model(num_classes: int = 2, **kwargs) -> SimplifiedDetector:
    """Factory function to create the simplified detector."""
    return SimplifiedDetector(num_classes=num_classes, **kwargs)
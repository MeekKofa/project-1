import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import resnet18, ResNet18_Weights

logger = logging.getLogger(__name__)

class SimplerAdaptiveFusionDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        logger.info("Initializing SimplerAdaptiveFusionDetector")
        
        # Load a simpler backbone - ResNet18
        try:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            logger.info("Loaded ResNet18 backbone successfully")
        except Exception as e:
            logger.error(f"Failed to load ResNet18 backbone: {e}")
            raise
            
        # Use the backbone layers we need
        backbone_dict = {
            'conv1': backbone.conv1,     # input processing
            'bn1': backbone.bn1,         # batch normalization
            'relu': backbone.relu,       # activation
            'maxpool': backbone.maxpool, # initial pooling
            'layer1': backbone.layer1,   # output 64 channels
            'layer2': backbone.layer2,   # output 128 channels
            'layer3': backbone.layer3,   # output 256 channels
            'layer4': backbone.layer4,   # output 512 channels
        }
        
        # Create feature extraction backbone
        backbone = nn.ModuleDict(backbone_dict)
        backbone.out_channels = 512  # ResNet18's last layer channels
        
        # Create FPN on top of backbone
        try:
            logger.info("Setting up Feature Pyramid Network")
            self.out_channels = 256
            self.fpn = torchvision.ops.FeaturePyramidNetwork(
                in_channels_list=[64, 128, 256, 512],
                out_channels=self.out_channels
            )
        except Exception as e:
            logger.error(f"Failed to create FPN: {e}")
            raise

        # Create anchor generator
        try:
            logger.info("Setting up anchor generator")
            # Match 4 FPN levels with appropriate anchor sizes
            anchor_generator = AnchorGenerator(
                sizes=tuple((s,) for s in [16, 32, 64, 128, 256]),  # Smaller anchors for better recall
                aspect_ratios=tuple((0.5, 0.75, 1.0, 1.5, 2.0) for _ in range(5))  # More diverse aspect ratios
            )
        except Exception as e:
            logger.error(f"Failed to create anchor generator: {e}")
            raise

        # Create ROI pooler
        try:
            logger.info("Setting up ROI pooler")
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2
            )
        except Exception as e:
            logger.error(f"Failed to create ROI pooler: {e}")
            raise

        # Create backbone with FPN
        try:
            logger.info("Creating backbone with FPN")
            backbone_with_fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
                backbone=backbone,
                return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
                in_channels_list=[64, 128, 256, 512],
                out_channels=256,
                extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelMaxPool()
            )
            logger.info("Successfully created backbone with FPN")
        except Exception as e:
            logger.error(f"Failed to create backbone with FPN: {e}")
            raise

        # Finally, create Faster R-CNN model
        try:
            logger.info("Creating Faster R-CNN model")
            self.model = FasterRCNN(
                backbone_with_fpn,
                num_classes=num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
            logger.info("Successfully created Faster R-CNN model")
        except Exception as e:
            logger.error(f"Failed to create Faster R-CNN model: {e}")
            raise

    def forward(self, images, targets=None):
        try:
            return self.model(images, targets)
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise

class AdaptiveFusionDetector(SimplerAdaptiveFusionDetector):
    """Alias for SimplerAdaptiveFusionDetector for compatibility"""
    pass
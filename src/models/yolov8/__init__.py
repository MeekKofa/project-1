"""
YOLOv8 model package.

Modular implementation of YOLOv8 with flexible backbone options:
- CSP: Original YOLOv8 backbone
- ResNet: ResNet18/34/50/101 with pretrained weights  
- ResNeXt: ResNeXt50/101 for better performance
"""

from .model import YOLOv8Model
from .loss import YOLOv8Loss
from .config import get_default_config, update_config
from .heads import YOLOv8Head, BoxRegressionHead, ClassificationHead, ObjectnessHead
from .backbones import (
    build_backbone,
    CSPBackbone,
    ResNetBackbone,
    ResNeXtBackbone
)

__all__ = [
    'YOLOv8Model',
    'YOLOv8Loss',
    'get_default_config',
    'update_config',
    'YOLOv8Head',
    'BoxRegressionHead',
    'ClassificationHead',
    'ObjectnessHead',
    'build_backbone',
    'CSPBackbone',
    'ResNetBackbone',
    'ResNeXtBackbone'
]

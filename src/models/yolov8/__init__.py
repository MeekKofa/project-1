"""
YOLOv8 model package.

Clean, modular implementation of YOLOv8 with ResNet18 backbone.
"""

from .architecture import YOLOv8Model
from .loss import YOLOv8Loss
from .config import get_default_config, update_config
from .heads import YOLOv8Head, BoxRegressionHead, ClassificationHead, ObjectnessHead

__all__ = [
    'YOLOv8Model',
    'YOLOv8Loss',
    'get_default_config',
    'update_config',
    'YOLOv8Head',
    'BoxRegressionHead',
    'ClassificationHead',
    'ObjectnessHead'
]

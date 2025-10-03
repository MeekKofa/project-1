"""
Models package.

Central access point for all detection models.
"""

from .model_loader import load_model, list_available_models, get_model_config
from .yolov8 import YOLOv8Model

__all__ = [
    'load_model',
    'list_available_models',
    'get_model_config',
    'YOLOv8Model'
]

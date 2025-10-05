"""
Models package.

Central access point for all detection models.
"""

from .registry import MODEL_REGISTRY, get_model, get_model_info, list_models

__all__ = [
    'MODEL_REGISTRY',
    'get_model',
    'get_model_info',
    'list_models',
]

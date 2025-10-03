"""
Model Loader - Single Point of Truth.

Central module for loading all detection models through registry.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from ..core.registry import ModelRegistry

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load a detection model by name.

    This is the single point of truth for model loading.
    All models must be registered in ModelRegistry.

    Args:
        model_name: Name of model in registry (e.g., 'yolov8', 'faster_rcnn')
        config: Optional configuration dict
        checkpoint_path: Optional path to checkpoint to load
        device: Device to load model on

    Returns:
        Loaded model

    Example:
        >>> # Load YOLOv8 with default config
        >>> model = load_model('yolov8')

        >>> # Load with custom config
        >>> config = {'num_classes': 3, 'dropout': 0.5}
        >>> model = load_model('yolov8', config=config)

        >>> # Load from checkpoint
        >>> model = load_model('yolov8', checkpoint_path='model.pth')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Loading model: {model_name}")

    # Build model from registry
    if config is None:
        config = {}

    model = ModelRegistry.build(model_name, **config)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            logger.info("✓ Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    # Move to device
    model = model.to(device)

    # Log model info
    info = model.get_model_info()
    logger.info(f"Model: {info['name']}")
    logger.info(f"Parameters: {info['total_parameters']:,}")
    logger.info(f"Device: {device}")

    return model


def list_available_models() -> list:
    """
    List all available models in registry.

    Returns:
        List of model names
    """
    return ModelRegistry.list_registered()


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get default configuration for a model.

    Args:
        model_name: Name of model

    Returns:
        Default configuration dict
    """
    if model_name == 'yolov8':
        from ..models.yolov8 import get_default_config
        return get_default_config()
    else:
        logger.warning(f"No default config available for {model_name}")
        return {}

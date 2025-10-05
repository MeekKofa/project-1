"""
YOLOv8 configuration.

Provides default hyperparameters and model configuration.
"""

from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """
    Get default YOLOv8 configuration.

    Returns:
        Configuration dictionary
    """
    return {
        # Model architecture
        'backbone': 'resnet18',
        'num_classes': 2,
        'dropout': 0.3,

        # Loss weights
        'box_weight': 7.5,
        'cls_weight': 0.5,
        'obj_weight': 1.0,

        # Focal loss parameters
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,

        # IoU threshold for matching
        'iou_thresh': 0.45,

        # Training
        'batch_size': 4,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 150,
        'warmup_epochs': 10,

        # Optimizer
        'optimizer': 'adamw',
        'momentum': 0.937,

        # Scheduler
        'scheduler': 'cosine',
        'min_lr_ratio': 0.01,

        # Augmentation
        'flip_prob': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,

        # Detection
        'conf_threshold': 0.25,
        'iou_threshold': 0.45,
        'max_det': 300,
        'max_proposals': None,

        # Training stability
        'gradient_clip': 10.0,
        'accumulation_steps': 4,
        'amp': True,

        # Box prediction
        'box_clamp_min': -10.0,
        'box_clamp_max': 10.0,
        'size_clamp_min': -4.0,
        'size_clamp_max': 4.0,
    }


def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with custom values.

    Args:
        base_config: Base configuration
        updates: Dictionary of values to update

    Returns:
        Updated configuration
    """
    config = base_config.copy()
    config.update(updates)
    return config

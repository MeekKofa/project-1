# Faster R-CNN training parameters
FASTER_RCNN_PARAMS = {
    'batch_size': 4,           # Small batch for 4GB GPU
    'accumulation_steps': 4,   # Effective batch size = 16
    'learning_rate': 0.002,    # Increased LR for better convergence
    'momentum': 0.9,
    'weight_decay': 0.0005,    # Increased for better regularization
    'num_epochs': 200,         # More epochs for better convergence
    'warmup_epochs': 5,        # Longer warmup for stable training
    'img_size': 512,           # Higher resolution for better detection
    'clip_grad_norm': 1.0,     # Less aggressive clipping
    'step_size': 30,           # Slower LR decay
    'gamma': 0.5,              # Less aggressive LR reduction
    'roi_batch_size': 512,     # More ROI proposals per batch
    'roi_positive_fraction': 0.25  # Better balance of positive/negative samples
}

# Alternative optimized configurations for different scenarios
FASTER_RCNN_HIGH_PRECISION = {
    'batch_size': 2,           # Smaller batch, higher resolution
    'accumulation_steps': 8,   # Effective batch size = 16
    'learning_rate': 0.001,    # Conservative LR
    'momentum': 0.9,
    'weight_decay': 0.001,     # Strong regularization
    'num_epochs': 300,         # More training
    'warmup_epochs': 10,       # Long warmup
    'img_size': 640,           # High resolution
    'clip_grad_norm': 0.5,
    'step_size': 50,
    'gamma': 0.3,
    'roi_batch_size': 256,
    'roi_positive_fraction': 0.5
}

FASTER_RCNN_FAST_TRAINING = {
    'batch_size': 8,           # Larger batch if GPU allows
    'accumulation_steps': 2,   # Effective batch size = 16
    'learning_rate': 0.005,    # Higher LR for faster training
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'num_epochs': 100,
    'warmup_epochs': 3,
    'img_size': 384,
    'clip_grad_norm': 2.0,
    'step_size': 20,
    'gamma': 0.7,
    'roi_batch_size': 1024,
    'roi_positive_fraction': 0.25
}

# YOLOv8 training parameters
YOLOV8_PARAMS = {
    'batch_size': 2,  # Small batch for 4GB GPU
    'accumulation_steps': 8,  # Effective batch = 16
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_epochs': 100,
    'input_size': 384,
    'dropout': 0.1,
    'warmup_epochs': 3
}

# Fusion parameters
FUSION_PARAMS = {
    "conf_thresh": 0.5,
    "iou_thresh": 0.5
}


def get_hyperparameters(model_name, profile='default'):
    """
    Get hyperparameters for a specific model and profile.

    Args:
        model_name: 'faster_rcnn', 'yolov8', etc.
        profile: 'default', 'high_precision', 'fast_training'

    Returns:
        Dictionary of hyperparameters
    """
    if model_name == 'faster_rcnn':
        if profile == 'high_precision':
            return FASTER_RCNN_HIGH_PRECISION
        elif profile == 'fast_training':
            return FASTER_RCNN_FAST_TRAINING
        else:
            return FASTER_RCNN_PARAMS
    elif model_name == 'yolov8':
        return YOLOV8_PARAMS
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Data augmentation parameters for better training
DATA_AUG_PARAMS = {
    'horizontal_flip_prob': 0.5,
    'brightness_factor': 0.2,
    'contrast_factor': 0.2,
    'saturation_factor': 0.2,
    'rotation_degrees': 15,
    'scale_factor': 0.1,
    'mixup_alpha': 0.2,        # For advanced augmentation
    'cutmix_alpha': 1.0,       # For advanced augmentation
}

# Training optimization parameters
OPTIMIZATION_PARAMS = {
    'use_amp': True,           # Automatic Mixed Precision
    'gradient_accumulation': True,
    'early_stopping_patience': 25,
    'early_stopping_min_delta': 0.001,
    'save_best_only': True,
    'monitor_metric': 'mAP@0.5',
}

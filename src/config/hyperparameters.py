# VGG16-YOLOv8 training parameters
VGG16_YOLOV8_PARAMS = {
    'batch_size': 8,          # Base batch size
    'accumulation_steps': 2,   # Gradient accumulation steps
    'box_format': 'cxcywh',   # Center-x, center-y, width, height format
    'coord_normalizer': 448,   # Match input image size for better scaling
    'min_box_size': 0.01,     # Minimum box size relative to image size
    'max_box_aspect': 4.0,    # More reasonable aspect ratio limit
    'objectness_threshold': 0.1, # Lower threshold for initial testing
    'use_gradient_checkpointing': False,  # Gradient checkpointing for memory efficiency
    'use_grid_priors': True,  # Use grid-based anchors
    'use_focal_loss': True,   # Use focal loss for classification
    'learning_rate': 0.001    # Higher learning rate
}

# Faster R-CNN training parameters optimized for current dataset
FASTER_RCNN_PARAMS = {
    'batch_size': 8,           # Increased batch size for better statistics
    'accumulation_steps': 2,   # Maintain effective batch size = 16
    'learning_rate': 0.001,    # More conservative LR for stability
    'momentum': 0.9,
    'weight_decay': 0.0005,    # Slightly increased for better regularization
    'num_epochs': 200,         # Reduced epochs with better stopping criteria
    'warmup_epochs': 5,        # Adjusted for new batch size
    'img_size': 800,           # Increased for better detection of varying sizes
    'clip_grad_norm': 1.0,     # More conservative clipping
    'step_size': 30,           # More frequent LR adjustments
    'gamma': 0.8,              # Gentler LR reduction
    'roi_batch_size': 1024,    # More ROI proposals for better recall
    'roi_positive_fraction': 0.3,  # More positive samples for better recall
    'score_threshold': 0.3,    # Lower threshold for predictions
    'nms_threshold': 0.5       # Standard NMS threshold
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


# VGG16-YOLOv8 training parameters
VGG16_YOLOV8_PARAMS = {
    'batch_size': 4,           # Small batch for 4GB GPU
    'accumulation_steps': 4,   # Effective batch size = 16
    'learning_rate': 0.002,    # Higher LR for better recall
    'momentum': 0.9,
    'weight_decay': 0.0003,    # Reduced for less conservative predictions
    'num_epochs': 250,         # More epochs for better convergence
    'warmup_epochs': 10,       # Longer warmup for stable training
    'img_size': 640,           # Higher resolution for better recall
    'clip_grad_norm': 2.0,     # Less restrictive clipping
    'step_size': 40,           # Slower LR decay
    'gamma': 0.7,              # Less aggressive LR reduction
    'score_threshold': 0.3,    # Lower threshold for predictions
    'nms_threshold': 0.5       # Standard NMS threshold
}

def get_hyperparameters(model_name, profile='default'):
    """
    Get hyperparameters for a specific model and profile.

    Args:
        model_name: 'faster_rcnn', 'yolov8', 'vgg16_yolov8', etc.
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
    elif model_name == 'vgg16_yolov8':
        return VGG16_YOLOV8_PARAMS
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

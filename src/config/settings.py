"""
Central configuration file for the project.
Contains all paths, settings, and constants used throughout the application.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
# Go up from src/config/ to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = SRC_DIR / "config"
DATASET_DIR = PROJECT_ROOT / "dataset"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUT_LOGS_DIR = OUTPUTS_DIR / "logs"
OUTPUT_IMAGES_DIR = OUTPUTS_DIR / "images"
OUTPUT_RESULTS_DIR = OUTPUTS_DIR / "results"

# Dataset paths
DATASET_CATTLEFACE_DIR = DATASET_DIR / "cattleface"
DATASET_IMAGES_DIR = DATASET_CATTLEFACE_DIR / "CowfaceImage"
DATASET_ANNOTATIONS_DIR = DATASET_CATTLEFACE_DIR / "Annotation"

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_CATTLEFACE_DIR = PROCESSED_DATA_DIR / "cattleface"

# Training data paths
TRAIN_IMAGES_DIR = PROCESSED_CATTLEFACE_DIR / "train"
VAL_IMAGES_DIR = PROCESSED_CATTLEFACE_DIR / "val"
TEST_IMAGES_DIR = PROCESSED_CATTLEFACE_DIR / "test"

# Pre-trained model weights (now in outputs/models)
PRETRAINED_WEIGHTS_DIR = OUTPUT_MODELS_DIR
YOLO11_WEIGHTS = OUTPUT_MODELS_DIR / "yolo11n.pt"
YOLOV8_WEIGHTS = OUTPUT_MODELS_DIR / "yolov8n.pt"

# Data processing module (moved to src)
PROCESSING_MODULE_DIR = SRC_DIR / "processing"

# NOTE: Directories are created on-demand, not upfront
# This ensures we only create directories when actually needed

# Training configurations - ALL outputs go to outputs directory
TRAINING_CONFIGS = {
    "faster_rcnn": {
        "name": "Faster R-CNN",
        "description": "Faster R-CNN with ResNet-50 backbone for object detection",
        "module": "src.training.train_faster_rcnn",
        "output_dir": OUTPUT_MODELS_DIR / "faster_rcnn",
        "log_file": OUTPUT_LOGS_DIR / "faster_rcnn.log",
        "results_dir": OUTPUT_RESULTS_DIR / "faster_rcnn",
        "images_dir": OUTPUT_IMAGES_DIR / "faster_rcnn",
        "checkpoints_dir": OUTPUT_MODELS_DIR / "faster_rcnn" / "checkpoints",
    },
    "yolov8": {
        "name": "YOLOv8",
        "description": "YOLOv8 model for real-time object detection",
        "module": "src.training.train_yolov8",
        "output_dir": OUTPUT_MODELS_DIR / "yolov8",
        "log_file": OUTPUT_LOGS_DIR / "yolov8.log",
        "results_dir": OUTPUT_RESULTS_DIR / "yolov8",
        "images_dir": OUTPUT_IMAGES_DIR / "yolov8",
        "checkpoints_dir": OUTPUT_MODELS_DIR / "yolov8" / "checkpoints",
    },
    "ultralytics": {
        "name": "Ultralytics YOLO",
        "description": "Ultralytics YOLO implementation",
        "module": "src.training.train_ultralytics",
        "output_dir": OUTPUT_MODELS_DIR / "ultralytics",
        "log_file": OUTPUT_LOGS_DIR / "ultralytics.log",
        "results_dir": OUTPUT_RESULTS_DIR / "ultralytics",
        "images_dir": OUTPUT_IMAGES_DIR / "ultralytics",
        "checkpoints_dir": OUTPUT_MODELS_DIR / "ultralytics" / "checkpoints",
    }
}

# Model evaluation configurations
EVALUATION_CONFIGS = {
    "metrics": {
        "output_dir": OUTPUT_RESULTS_DIR / "metrics",
        "log_file": OUTPUT_LOGS_DIR / "evaluation.log",
    },
    "visualization": {
        "output_dir": OUTPUT_IMAGES_DIR / "evaluations",
        "log_file": OUTPUT_LOGS_DIR / "visualization.log",
    }
}

# Default hyperparameters
DEFAULT_HYPERPARAMETERS = {
    "batch_size": 16,
    "learning_rate": 0.001,
    "epochs": 100,
    "image_size": 640,
    "workers": 4,
}


def get_config(config_type: str = "training") -> Dict[str, Any]:
    """Get configuration dictionary for specified type."""
    if config_type == "training":
        return TRAINING_CONFIGS
    elif config_type == "evaluation":
        return EVALUATION_CONFIGS
    elif config_type == "paths":
        return {
            "project_root": PROJECT_ROOT,
            "src_dir": SRC_DIR,
            "outputs_dir": OUTPUTS_DIR,
            "processing_dir": PROCESSING_MODULE_DIR,
            "dataset_dir": DATASET_DIR,
        }
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def ensure_output_dirs():
    """
    DEPRECATED: Directory creation is now on-demand.
    This function is kept for backward compatibility but does nothing.
    """
    # Directories are now created only when needed
    pass


def create_output_dir(path: Path) -> Path:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_systematic_output_dir(dataset_name: str, model_name: str, output_type: str) -> Path:
    """
    Get systematic output directory path following the pattern:
    outputs/{dataset}/{model}/{output_type}/

    NOTE: This now returns the path without creating the directory.
    Directories are created on-demand when files are actually saved.

    Args:
        dataset_name: Name of the dataset (e.g., cattlebody, cattleface)
        model_name: Name of the model (faster_rcnn, yolov8, ultralytics)
        output_type: Type of output (models, logs, results, images, metrics, checkpoints)

    Returns:
        Path object for the systematic output directory (not created yet)
    """
    base_output_dir = OUTPUTS_DIR / dataset_name / model_name / output_type
    return base_output_dir  # Return path without creating directory


def ensure_dir(path: Path) -> Path:
    """Create directory only when needed - used for on-demand creation."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_legacy_output_dir(model_name: str, output_type: str) -> Path:
    """
    Legacy output directory function for backward compatibility.

    Args:
        model_name: Name of the model (faster_rcnn, yolov8, ultralytics)
        output_type: Type of output (models, logs, results, images, checkpoints)

    Returns:
        Path object for the output directory (created on-demand)
    """
    output_path = None

    if model_name in TRAINING_CONFIGS:
        if output_type == "models":
            output_path = TRAINING_CONFIGS[model_name]["output_dir"]
        elif output_type == "logs":
            output_path = TRAINING_CONFIGS[model_name]["log_file"].parent
        elif output_type == "results":
            output_path = TRAINING_CONFIGS[model_name]["results_dir"]
        elif output_type == "images":
            output_path = TRAINING_CONFIGS[model_name]["images_dir"]
        elif output_type == "checkpoints":
            output_path = TRAINING_CONFIGS[model_name]["checkpoints_dir"]

    # Fallback to general output directories
    if output_path is None:
        if output_type == "models":
            output_path = OUTPUT_MODELS_DIR
        elif output_type == "logs":
            output_path = OUTPUT_LOGS_DIR
        elif output_type == "results":
            output_path = OUTPUT_RESULTS_DIR
        elif output_type == "images":
            output_path = OUTPUT_IMAGES_DIR
        else:
            output_path = OUTPUTS_DIR / output_type

    return output_path  # Return path without creating directory


# Alias for backward compatibility
get_output_dir = get_legacy_output_dir
get_output_path = get_legacy_output_dir  # Another alias for compatibility


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get complete configuration for a specific model."""
    if model_name not in TRAINING_CONFIGS:
        raise ValueError(
            f"Model '{model_name}' not found in training configurations")
    return TRAINING_CONFIGS[model_name]


def enforce_output_structure():
    """
    Enforce the output directory structure.
    This function should be called to ensure no outputs leak outside outputs/ directory.
    """
    import warnings

    # Check if any legacy directories exist in project root
    legacy_dirs = ["runs", "weights", "image_out", "gradcam"]
    for legacy_dir in legacy_dirs:
        legacy_path = PROJECT_ROOT / legacy_dir
        if legacy_path.exists():
            warnings.warn(
                f"Legacy directory '{legacy_dir}' found in project root. "
                f"Please move contents to appropriate outputs/ subdirectory."
            )


# Only enforce structure check on import (no automatic directory creation)
enforce_output_structure()

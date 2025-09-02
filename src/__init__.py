"""
Cattle Detection and Recognition System - Source Package

This package contains all the source code for the cattle detection and recognition system,
including models, training scripts, utilities, and configuration.
"""

__version__ = "1.0.0"
__author__ = "Cattle Detection Team"
__email__ = "team@cattle-detection.ai"

# Make key components easily accessible
from .config import (
    TRAINING_CONFIGS,
    EVALUATION_CONFIGS,
    OUTPUT_MODELS_DIR,
    OUTPUT_LOGS_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_RESULTS_DIR,
    get_config,
    ensure_output_dirs
)

__all__ = [
    "TRAINING_CONFIGS",
    "EVALUATION_CONFIGS",
    "OUTPUT_MODELS_DIR",
    "OUTPUT_LOGS_DIR",
    "OUTPUT_IMAGES_DIR",
    "OUTPUT_RESULTS_DIR",
    "get_config",
    "ensure_output_dirs"
]

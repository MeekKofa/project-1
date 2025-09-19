import os
from pathlib import Path

# Base paths - adjusted to work with our current structure
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up to project root
DATASET_ROOT = PROJECT_ROOT / "dataset" / "cattleface"

# Use our processed data structure instead of the hardcoded Windows path
PROCESSED_DATA_ROOT = PROJECT_ROOT / "processed_data" / "cattleface"

# Dataset organization paths
TRAIN_IMAGES = PROCESSED_DATA_ROOT / 'train'
# Labels are in the same directory for our structure
TRAIN_LABELS = PROCESSED_DATA_ROOT / 'train'
VAL_IMAGES = PROCESSED_DATA_ROOT / 'val'
VAL_LABELS = PROCESSED_DATA_ROOT / 'val'

# Model paths - use our outputs structure
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
YOLOV8_PATH = MODELS_DIR / 'yolov8.pth'
FASTER_RCNN_PATH = MODELS_DIR / 'faster_rcnn.pth'

# Convert paths to strings for backward compatibility
TRAIN_IMAGES = str(TRAIN_IMAGES)
TRAIN_LABELS = str(TRAIN_LABELS)
VAL_IMAGES = str(VAL_IMAGES)
VAL_LABELS = str(VAL_LABELS)
YOLOV8_PATH = str(YOLOV8_PATH)
FASTER_RCNN_PATH = str(FASTER_RCNN_PATH)
MODELS_DIR = str(MODELS_DIR)

# Verify paths exist (only show warnings for actual dataset paths)
dataset_paths = [TRAIN_IMAGES, VAL_IMAGES]
for path in dataset_paths:
    if not os.path.exists(path):
        print(f"Info: Dataset path does not exist yet: {path}")

# Note: Output directories are now created on-demand when files are saved
# This eliminates empty directory creation during import

from ultralytics import YOLO
import torch
from pathlib import Path
import sys
import os
from src.config.hyperparameters import YOLOV8_PARAMS


def verify_paths():
    try:
        # Get script location and project root
        if '__file__' in globals():
            script_path = Path(__file__).resolve()
            project_root = script_path.parent.parent
        else:
            # Handle running from different directory
            project_root = Path.cwd()
            while not (project_root / 'cattle-dataset').exists():
                if project_root == project_root.parent:
                    raise FileNotFoundError("Could not find project root")
                project_root = project_root.parent

        print(f"Found project root at: {project_root}")
        return project_root
    except Exception as e:
        print(f"Error finding paths: {e}")
        sys.exit(1)


def train_yolo(**kwargs):
    try:
        # Verify and set project directory
        PROJ_DIR = verify_paths()
        DATA_YAML = PROJ_DIR / 'cattle-dataset/cattlebody/data.yaml'

        print(f"Project directory: {PROJ_DIR}")
        print(f"Data YAML path: {DATA_YAML}")
        print(f"Original working directory: {os.getcwd()}")

        # Change to project directory
        os.chdir(PROJ_DIR)
        print(f"Changed working directory to: {os.getcwd()}")

        # Get optimized parameters from config
        IMG_SIZE = YOLOV8_PARAMS['input_size']
        BATCH_SIZE = kwargs.get('batch_size', YOLOV8_PARAMS['batch_size'])
        EPOCHS = kwargs.get('epochs', YOLOV8_PARAMS['num_epochs'])
        LR0 = YOLOV8_PARAMS['learning_rate']
        OPTIMIZER = YOLOV8_PARAMS.get('optimizer', 'AdamW')
        MOMENTUM = YOLOV8_PARAMS.get('momentum', 0.937)
        WEIGHT_DECAY = YOLOV8_PARAMS['weight_decay']
        WARMUP_EPOCHS = YOLOV8_PARAMS['warmup_epochs']
        PATIENCE = YOLOV8_PARAMS.get('patience', 50)
        AMP = YOLOV8_PARAMS.get('amp', True)

        print(f"Using optimized parameters:")
        print(f"  Image size: {IMG_SIZE}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Epochs: {EPOCHS}")
        print(f"  Learning rate: {LR0}")
        print(f"  Optimizer: {OPTIMIZER}")
        print(f"  AMP: {AMP}")

        # Initialize YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Training settings optimized for cattle detection
        results = model.train(
            data=str(DATA_YAML),
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            epochs=EPOCHS,
            patience=PATIENCE,
            device=0,
            amp=AMP,
            optimizer=OPTIMIZER,
            lr0=LR0,
            lrf=1e-4,  # Final learning rate
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            warmup_epochs=WARMUP_EPOCHS,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # Enhanced data augmentation for cattle detection
            hsv_h=YOLOV8_PARAMS.get('hsv_h', 0.015),
            hsv_s=YOLOV8_PARAMS.get('hsv_s', 0.7),
            hsv_v=YOLOV8_PARAMS.get('hsv_v', 0.4),
            degrees=YOLOV8_PARAMS.get('degrees', 0.0),
            translate=YOLOV8_PARAMS.get('translate', 0.1),
            scale=YOLOV8_PARAMS.get('scale', 0.5),
            shear=YOLOV8_PARAMS.get('shear', 0.0),
            perspective=YOLOV8_PARAMS.get('perspective', 0.0),
            flipud=YOLOV8_PARAMS.get('flipud', 0.0),
            fliplr=YOLOV8_PARAMS.get('fliplr', 0.5),
            mosaic=YOLOV8_PARAMS.get('mosaic', 1.0),
            mixup=YOLOV8_PARAMS.get('mixup', 0.1),
            save=True,
            cache=False,
            project=YOLOV8_PARAMS.get('project', 'cattle_detection'),
            name=YOLOV8_PARAMS.get('name', 'yolov8_resnet18')
        )

    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)


def main(**kwargs):
    """
    Main function called by the CLI training system.

    Args:
        **kwargs: Training arguments passed from main.py

    Returns:
        bool: True if training succeeded, False otherwise
    """
    try:
        # Call train_yolo with the passed arguments
        train_yolo(**kwargs)
        return True
    except Exception as e:
        print(f"❌ Ultralytics YOLO training failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Starting YOLOv8 training script...")
    train_yolo()

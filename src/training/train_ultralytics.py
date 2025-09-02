from ultralytics import YOLO
import torch
from pathlib import Path
import sys
import os


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

        # Configuration optimized for RTX 3050 4GB
        IMG_SIZE = 416
        BATCH_SIZE = kwargs.get('batch_size', 8)
        EPOCHS = kwargs.get('epochs', 100)

        # Initialize YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Training settings optimized for RTX 3050 4GB
        results = model.train(
            data=str(DATA_YAML),
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            epochs=EPOCHS,
            patience=20,
            device=0,
            amp=True,
            optimizer='AdamW',
            lr0=1e-3,
            lrf=1e-4,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            save=True,
            cache=False
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

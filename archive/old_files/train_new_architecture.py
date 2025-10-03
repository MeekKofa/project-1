"""
Example: Training YOLOv8 with New Modular Architecture.

This demonstrates how to use the new registry-based system.
"""

from src.training.trainer import create_trainer
from src.data import DetectionDataset, create_detection_dataloaders
from src.models import load_model, list_available_models, get_model_config
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from new modular architecture


def main():
    """Main training function."""

    # 1. List available models
    logger.info("Available models:")
    for model_name in list_available_models():
        logger.info(f"  - {model_name}")

    # 2. Setup paths
    project_root = Path(__file__).parent
    train_images = project_root / "dataset/cattlebody/train/images"
    train_labels = project_root / "dataset/cattlebody/train/labels"
    val_images = project_root / "dataset/cattlebody/val/images"
    val_labels = project_root / "dataset/cattlebody/val/labels"
    output_dir = project_root / "outputs/new_architecture/yolov8"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Configuration
    config = {
        'num_classes': 2,
        'learning_rate': 1e-3,
        'batch_size': 8,
        'num_epochs': 50,
        'image_size': 640,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    logger.info(f"Training on device: {config['device']}")

    # 4. Create dataloaders (universal dataset)
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_detection_dataloaders(
        train_images_dir=str(train_images),
        train_labels_dir=str(train_labels),
        val_images_dir=str(val_images),
        val_labels_dir=str(val_labels),
        batch_size=config['batch_size'],
        num_workers=4,
        image_size=config['image_size']
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # 5. Load model through registry (single point of truth)
    logger.info("Loading model...")
    model_config = {
        'num_classes': config['num_classes']
    }
    model = load_model(
        model_name='yolov8',
        config=model_config,
        device=torch.device(config['device'])
    )

    # 6. Create trainer (universal trainer)
    logger.info("Creating trainer...")
    trainer = create_trainer(
        model_name='yolov8',
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device(config['device']),
        output_dir=str(output_dir),
        learning_rate=config['learning_rate'],
        weight_decay=5e-4,
        grad_clip=10.0
    )

    # 7. Train
    logger.info("\n" + "="*60)
    logger.info("Starting Training")
    logger.info("="*60 + "\n")

    history = trainer.train(num_epochs=config['num_epochs'])

    # 8. Report results
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Best mAP: {history['best_mAP']:.4f}")
    logger.info(f"Best Epoch: {history['best_epoch'] + 1}")
    logger.info(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()

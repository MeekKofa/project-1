"""
Test Script: Validate New Architecture.

Quick tests to ensure everything is working.
"""

import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_registry():
    """Test registry system."""
    logger.info("\n" + "="*60)
    logger.info("Testing Registry System")
    logger.info("="*60)

    from src.core.registry import ModelRegistry

    # Check registered models
    models = ModelRegistry.list_registered()
    logger.info(f"✓ Registered models: {models}")

    assert 'yolov8' in models, "YOLOv8 not registered!"
    logger.info("✓ YOLOv8 is registered")


def test_model_loading():
    """Test model loading through registry."""
    logger.info("\n" + "="*60)
    logger.info("Testing Model Loading")
    logger.info("="*60)

    from src.models import load_model, list_available_models

    # List models
    available = list_available_models()
    logger.info(f"✓ Available models: {available}")

    # Load YOLOv8
    model = load_model('yolov8', config={'num_classes': 2})
    logger.info(f"✓ Model loaded: {model.__class__.__name__}")

    # Check model info
    info = model.get_model_info()
    logger.info(
        f"✓ Model info: {info['name']}, {info['total_parameters']:,} params")


def test_model_forward():
    """Test model forward pass."""
    logger.info("\n" + "="*60)
    logger.info("Testing Model Forward Pass")
    logger.info("="*60)

    from src.models import load_model

    # Load model
    model = load_model('yolov8', config={'num_classes': 2})
    model.eval()

    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 640, 640)

    # Forward pass
    with torch.no_grad():
        box_preds, cls_preds, obj_preds = model(images)

    logger.info(f"✓ Input shape: {images.shape}")
    logger.info(f"✓ Box predictions: {box_preds.shape}")
    logger.info(f"✓ Class predictions: {cls_preds.shape}")
    logger.info(f"✓ Objectness predictions: {obj_preds.shape}")

    assert box_preds.shape[0] == batch_size
    assert cls_preds.shape[0] == batch_size
    assert obj_preds.shape[0] == batch_size
    logger.info("✓ Shapes are correct!")


def test_loss_computation():
    """Test loss computation."""
    logger.info("\n" + "="*60)
    logger.info("Testing Loss Computation")
    logger.info("="*60)

    from src.models import load_model

    # Load model
    model = load_model('yolov8', config={'num_classes': 2})
    model.train()

    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 640, 640)

    # Create dummy targets
    targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),
            'labels': torch.tensor([0, 1], dtype=torch.int64)
        },
        {
            'boxes': torch.tensor([[150, 150, 250, 250]], dtype=torch.float32),
            'labels': torch.tensor([0], dtype=torch.int64)
        }
    ]

    # Forward pass
    predictions = model(images, targets)

    # Compute loss
    loss = model.compute_loss(predictions, targets)

    logger.info(f"✓ Loss computed: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive!"
    logger.info("✓ Loss is valid!")


def test_dataset():
    """Test universal dataset."""
    logger.info("\n" + "="*60)
    logger.info("Testing Universal Dataset")
    logger.info("="*60)

    from src.data import DetectionDataset

    # Check if dataset exists
    project_root = Path(__file__).parent
    train_images = project_root / "dataset/cattlebody/train/images"
    train_labels = project_root / "dataset/cattlebody/train/labels"

    if not train_images.exists():
        logger.warning("⚠ Dataset not found, skipping test")
        return

    # Create dataset
    dataset = DetectionDataset(
        images_dir=str(train_images),
        labels_dir=str(train_labels),
        image_size=640,
        augment=False
    )

    logger.info(f"✓ Dataset created: {len(dataset)} samples")

    # Get one sample
    image, target = dataset[0]
    logger.info(f"✓ Image shape: {image.shape}")
    logger.info(f"✓ Boxes: {target['boxes'].shape}")
    logger.info(f"✓ Labels: {target['labels'].shape}")


def test_integration():
    """Test full integration."""
    logger.info("\n" + "="*60)
    logger.info("Testing Full Integration")
    logger.info("="*60)

    from src.models import load_model
    from src.data import DetectionDataset
    from torch.utils.data import DataLoader

    # Check dataset
    project_root = Path(__file__).parent
    train_images = project_root / "dataset/cattlebody/train/images"
    train_labels = project_root / "dataset/cattlebody/train/labels"

    if not train_images.exists():
        logger.warning("⚠ Dataset not found, skipping test")
        return

    # Create dataset and loader
    dataset = DetectionDataset(
        images_dir=str(train_images),
        labels_dir=str(train_labels),
        image_size=640,
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=DetectionDataset.collate_fn
    )

    # Load model
    model = load_model('yolov8', config={'num_classes': 2})
    model.train()

    # Get one batch
    images, targets = next(iter(loader))
    logger.info(f"✓ Batch loaded: {images.shape}")

    # Forward pass
    predictions = model(images, targets)
    logger.info(f"✓ Forward pass successful")

    # Compute loss
    loss = model.compute_loss(predictions, targets)
    logger.info(f"✓ Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    logger.info(f"✓ Backward pass successful")

    logger.info("✓ Full integration working!")


def main():
    """Run all tests."""
    logger.info("\n" + "#"*60)
    logger.info("# New Architecture Validation Tests")
    logger.info("#"*60)

    try:
        test_registry()
        test_model_loading()
        test_model_forward()
        test_loss_computation()
        test_dataset()
        test_integration()

        logger.info("\n" + "="*60)
        logger.info("✅ All Tests Passed!")
        logger.info("="*60 + "\n")

    except Exception as e:
        logger.error(f"\n❌ Test Failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Quick validation script to verify YOLOv8 training setup is correct.
Run this before starting full training to catch issues early.
"""

from torchvision import transforms
from torch.utils.data import DataLoader
from src.config.hyperparameters import YOLOV8_PARAMS
from src.models.yolov8 import ResNet18_YOLOv8, Dataset as CattleDataset
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def validate_model_setup():
    """Validate model initialization and basic forward pass."""
    print("=" * 70)
    print("YOLOv8 Training Setup Validation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")

    # Test model initialization
    print("\n[1/5] Testing model initialization...")
    try:
        model = ResNet18_YOLOv8(
            num_classes=2,
            dropout=YOLOV8_PARAMS['dropout'],
            box_weight=YOLOV8_PARAMS.get('box_loss_weight', 7.5),
            cls_weight=YOLOV8_PARAMS.get('cls_loss_weight', 0.5)
        )
        model.to(device)
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Box loss weight: {model.criterion.box_weight}")
        print(f"  ✓ Cls loss weight: {model.criterion.cls_weight}")
        print(f"  ✓ Focal gamma: {model.criterion.gamma}")
    except Exception as e:
        print(f"  ✗ Model initialization failed: {e}")
        return False

    # Test forward pass
    print("\n[2/5] Testing forward pass...")
    try:
        dummy_input = torch.randn(2, 3, 640, 640).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)

        # Check inference mode output
        assert isinstance(outputs, list), "Inference should return list"
        assert len(outputs) == 2, "Should have 2 batch items"
        assert 'boxes' in outputs[0], "Should have boxes key"
        assert 'scores' in outputs[0], "Should have scores key"
        assert 'labels' in outputs[0], "Should have labels key"
        print(f"  ✓ Inference mode works correctly")
        print(f"  ✓ Output format: list of dicts with boxes/scores/labels")

        # Check training mode output
        model.train()
        dummy_targets = {
            'boxes': [torch.tensor([[10, 10, 100, 100]], device=device, dtype=torch.float32) for _ in range(2)],
            'labels': [torch.tensor([0], device=device, dtype=torch.long) for _ in range(2)]
        }
        outputs = model(dummy_input, dummy_targets)
        assert isinstance(outputs, tuple), "Training should return tuple"
        assert len(outputs) == 2, "Should return (box_preds, cls_preds)"
        box_preds, cls_preds = outputs
        print(f"  ✓ Training mode works correctly")
        print(f"  ✓ Box preds shape: {box_preds.shape}")
        print(f"  ✓ Cls preds shape: {cls_preds.shape}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test loss computation
    print("\n[3/5] Testing loss computation...")
    try:
        model.train()
        loss = model.compute_loss(box_preds, cls_preds, dummy_targets)
        assert loss is not None, "Loss should not be None"
        assert torch.isfinite(loss), "Loss should be finite"
        print(f"  ✓ Loss computed successfully: {loss.item():.4f}")
        print(f"  ✓ Loss is finite: {torch.isfinite(loss).item()}")
    except Exception as e:
        print(f"  ✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test box prediction constraints
    print("\n[4/5] Testing box prediction constraints...")
    try:
        # Create extreme values to test clamping
        extreme_input = torch.randn(1, 3, 640, 640).to(
            device) * 100  # Very large values
        model.eval()
        with torch.no_grad():
            outputs = model(extreme_input)

        boxes = outputs[0]['boxes']
        # Check boxes are reasonable (not exploded)
        assert boxes.shape[1] == 4, "Boxes should have 4 coordinates"
        if boxes.numel() > 0:
            max_coord = boxes.abs().max().item()
            assert max_coord < 10000, f"Box coordinates too large: {max_coord}"
            print(
                f"  ✓ Box predictions constrained (max coord: {max_coord:.1f})")
        else:
            print(f"  ✓ No predictions (acceptable for random input)")
    except Exception as e:
        print(f"  ✗ Box constraint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test dataset loading
    print("\n[5/5] Testing dataset loading...")
    try:
        train_images_dir = "processed_data/cattle/train/images"
        train_labels_dir = "processed_data/cattle/train/labels"

        if not os.path.exists(train_images_dir):
            print(f"  ⚠ Dataset not found at {train_images_dir}")
            print(
                f"  ⚠ Run preprocessing first: python main.py preprocess -d cattle -s 0.8 -f")
            return None  # Not a failure, just not ready

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        dataset = CattleDataset(
            train_images_dir,
            train_labels_dir,
            transform=transform,
            target_size=(640, 640),
            augment=False
        )

        print(f"  ✓ Dataset loaded: {len(dataset)} images")

        # Test single sample
        img, target = dataset[0]
        assert img.shape == (
            3, 640, 640), f"Image shape should be (3, 640, 640), got {img.shape}"
        assert 'boxes' in target, "Target should have boxes"
        assert 'labels' in target, "Target should have labels"
        print(
            f"  ✓ Sample loaded: image {img.shape}, {len(target['boxes'])} boxes")

        # Test dataloader
        def collate_fn(batch):
            images = torch.stack([x[0] for x in batch])
            targets = [x[1] for x in batch]
            return images, targets

        loader = DataLoader(dataset, batch_size=2,
                            shuffle=False, collate_fn=collate_fn)
        images, targets = next(iter(loader))
        assert images.shape == (2, 3, 640, 640), "Batch shape incorrect"
        print(f"  ✓ DataLoader works: batch {images.shape}")

    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # All tests passed
    print("\n" + "=" * 70)
    print("✅ ALL VALIDATION CHECKS PASSED!")
    print("=" * 70)
    print("\nYou can now start training with confidence:")
    print("  python main.py train -m yolov8 -d cattle -e 10 -b 4 --device cuda:1")
    print("\nExpected results after 10 epochs:")
    print("  - Loss should decrease: 2.2 → 1.5 → 1.2")
    print("  - mAP should increase: 15% → 30% → 45%")
    print("=" * 70)
    return True


if __name__ == "__main__":
    result = validate_model_setup()
    if result is False:
        print("\n❌ Validation failed! Fix issues before training.")
        sys.exit(1)
    elif result is None:
        print("\n⚠️  Validation incomplete (dataset not found).")
        sys.exit(2)
    else:
        print("\n✅ Ready for training!")
        sys.exit(0)

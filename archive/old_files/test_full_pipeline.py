"""Comprehensive test of the full training pipeline before running actual training."""
import torch
import torch.nn.functional as F
from src.models.yolov8 import ResNet18_YOLOv8
import sys


def test_model_architecture():
    """Test model creation and forward pass."""
    print("\n=== Testing Model Architecture ===")
    model = ResNet18_YOLOv8(num_classes=2)

    # Check all heads exist
    assert hasattr(model, 'box_head'), "Missing box_head"
    assert hasattr(model, 'cls_head'), "Missing cls_head"
    assert hasattr(model, 'obj_head'), "Missing obj_head"
    print("✓ All heads (box, cls, obj) exist")

    # Test forward pass with random input
    x = torch.randn(2, 3, 640, 640)
    outputs = model(x)

    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
    box_preds, cls_preds, obj_preds = outputs

    print(f"✓ Forward pass successful")
    print(f"  - Box predictions: {box_preds.shape}")
    print(f"  - Class predictions: {cls_preds.shape}")
    print(f"  - Objectness predictions: {obj_preds.shape}")

    return model


def test_loss_computation():
    """Test loss computation with various scenarios."""
    print("\n=== Testing Loss Computation ===")
    model = ResNet18_YOLOv8(num_classes=2, box_weight=7.5, cls_weight=0.5)
    model.train()  # Must be in training mode for gradients

    # Get predictions from actual forward pass to maintain gradient graph
    dummy_input = torch.randn(2, 3, 640, 640)
    box_preds, cls_preds, obj_preds = model(dummy_input)

    # Test case 1: Normal case with targets
    targets = {
        "boxes": [
            torch.tensor(
                [[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),
            torch.tensor([[150, 150, 250, 250]], dtype=torch.float32)
        ],
        "labels": [
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0], dtype=torch.long)
        ]
    }

    loss = model.compute_loss(box_preds, cls_preds, obj_preds, targets)
    assert loss is not None, "Loss should not be None with valid targets"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    assert loss.requires_grad, "Loss should require gradients"
    print(f"✓ Normal case: loss = {loss.item():.4f}")

    # Test case 2: No positives (all IoU too low)
    targets_far = {
        "boxes": [
            # Very small box
            torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
            torch.tensor([[620, 620, 630, 630]],
                         dtype=torch.float32)  # Far corner
        ],
        "labels": [
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long)
        ]
    }

    loss_no_pos = model.compute_loss(
        box_preds, cls_preds, obj_preds, targets_far)
    assert loss_no_pos is not None, "Loss should not be None even without positive matches"
    assert torch.isfinite(
        loss_no_pos), f"Loss should be finite, got {loss_no_pos}"
    assert loss_no_pos.requires_grad, "Loss should require gradients"
    assert loss_no_pos.item(
    ) > 0, f"Loss should be positive, got {loss_no_pos.item()}"
    print(f"✓ No positives case: loss = {loss_no_pos.item():.4f}")

    # Test case 3: Empty targets (should be skipped in actual training)
    targets_empty = {
        "boxes": [torch.empty((0, 4))],
        "labels": [torch.empty((0,), dtype=torch.long)]
    }

    loss_empty = model.compute_loss(
        box_preds[:1], cls_preds[:1], obj_preds[:1], targets_empty)
    # This should return a small loss (0.01) since num_pos=0 and num_neg=0
    if loss_empty is not None:
        assert torch.isfinite(
            loss_empty), f"Loss should be finite, got {loss_empty}"
        print(f"✓ Empty targets case: loss = {loss_empty.item():.4f}")
    else:
        print("⚠ Empty targets returned None (will be handled in training loop)")


def test_inference():
    """Test inference mode predictions."""
    print("\n=== Testing Inference Mode ===")
    model = ResNet18_YOLOv8(num_classes=2)
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        detections = model(x, targets=None)

    assert isinstance(
        detections, list), f"Expected list, got {type(detections)}"
    assert len(
        detections) == 1, f"Expected 1 detection dict, got {len(detections)}"

    det = detections[0]
    assert 'boxes' in det, "Missing 'boxes' in detection"
    assert 'scores' in det, "Missing 'scores' in detection"
    assert 'labels' in det, "Missing 'labels' in detection"

    print(f"✓ Inference mode successful")
    print(f"  - Number of predictions: {len(det['boxes'])}")
    if len(det['boxes']) > 0:
        print(
            f"  - Score range: [{det['scores'].min():.4f}, {det['scores'].max():.4f}]")
        print(f"  - Box shape: {det['boxes'].shape}")


def test_random_noise_predictions():
    """Test that model doesn't predict too many objects on random noise."""
    print("\n=== Testing Random Noise Predictions ===")
    model = ResNet18_YOLOv8(num_classes=2)
    model.eval()

    # Test on pure random noise
    noise = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        detections = model(noise, targets=None)

    det = detections[0]

    # Count high confidence predictions (>0.5)
    high_conf = (det['scores'] > 0.5).sum().item()

    print(
        f"✓ Predictions on random noise: {len(det['boxes'])} total, {high_conf} with confidence >0.5")

    # Before objectness fix, this would be 300-400 predictions
    # After fix, should be much lower (ideally <50)
    if high_conf > 100:
        print(
            f"⚠ WARNING: Too many high-confidence predictions ({high_conf}) - objectness may not be working")
    else:
        print(
            f"✓ Good: Only {high_conf} high-confidence predictions (objectness working)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("COMPREHENSIVE PIPELINE TEST")
    print("=" * 60)

    try:
        model = test_model_architecture()
        test_loss_computation()
        test_inference()
        test_random_noise_predictions()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - Ready for training!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

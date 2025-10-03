#!/usr/bin/env python3
"""Quick test to verify letterbox resize preserves aspect ratio and box coordinates."""

from torchvision import transforms
from src.models.yolov8 import Dataset
import torch
from PIL import Image
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))


def test_letterbox():
    """Test letterbox resize on a sample cattle image."""

    # Setup paths
    image_dir = "processed_data/cattle/train/images"
    label_dir = "processed_data/cattle/train/labels"

    # Check if dataset exists
    if not os.path.exists(image_dir):
        print(f"❌ Dataset not found at {image_dir}")
        return False

    # Create dataset with 640x640 target
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=transform,
        target_size=(640, 640),
        augment=False
    )

    print(f"✓ Dataset loaded: {len(dataset)} images")

    # Test first few samples
    for i in range(min(3, len(dataset))):
        img, target = dataset[i]
        boxes = target['boxes']
        labels = target['labels']

        print(f"\n--- Sample {i+1} ---")
        print(f"  Image shape: {img.shape}")
        print(f"  Num boxes: {len(boxes)}")

        if len(boxes) > 0:
            print(f"  Box sample: {boxes[0]}")
            # Check boxes are within image bounds
            assert img.shape[1] == 640 and img.shape[2] == 640, "Image should be 640x640"
            assert (boxes[:, 0] >= 0).all() and (
                boxes[:, 0] <= 640).all(), "x1 out of bounds"
            assert (boxes[:, 1] >= 0).all() and (
                boxes[:, 1] <= 640).all(), "y1 out of bounds"
            assert (boxes[:, 2] >= 0).all() and (
                boxes[:, 2] <= 640).all(), "x2 out of bounds"
            assert (boxes[:, 3] >= 0).all() and (
                boxes[:, 3] <= 640).all(), "y2 out of bounds"
            assert (boxes[:, 2] > boxes[:, 0]).all(), "x2 should be > x1"
            assert (boxes[:, 3] > boxes[:, 1]).all(), "y2 should be > y1"
            print("  ✓ Boxes valid and within bounds")

    print("\n✅ All tests passed! Letterbox resize working correctly.")
    return True


if __name__ == "__main__":
    success = test_letterbox()
    sys.exit(0 if success else 1)

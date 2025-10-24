"""
Test script to verify visualization score display logic.
"""

import torch
from PIL import Image, ImageDraw
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("TESTING VISUALIZATION SCORE DISPLAY LOGIC")
print("="*70)

# Test data
test_predictions = {
    'boxes': torch.tensor([[10, 10, 100, 100], [150, 150, 250, 250]]),
    'scores': torch.tensor([0.12, 0.89]),
    'labels': torch.tensor([0, 0])
}

class_names = ['Cattlebody']
pred_conf_threshold = 0.5

print(f"\nTest Predictions:")
print(f"  Boxes: {test_predictions['boxes'].shape}")
print(f"  Scores: {test_predictions['scores'].tolist()}")
print(f"  Labels: {test_predictions['labels'].tolist()}")
print(f"  Confidence threshold: {pred_conf_threshold}")

print(f"\nFiltering logic:")
for idx, (box, score, label) in enumerate(zip(
    test_predictions['boxes'],
    test_predictions['scores'],
    test_predictions['labels']
)):
    score_val = float(score)
    print(f"  Prediction {idx}: score={score_val:.2f} ", end="")

    if score_val < pred_conf_threshold:
        print(f"❌ FILTERED (below {pred_conf_threshold})")
    else:
        print(f"✓ SHOWN")
        label_name = class_names[label] if label < len(
            class_names) else f'class_{label}'
        label_text = f"{label_name}: {score_val:.2f}"
        print(f"    Label would be: '{label_text}'")

print(f"\n{'='*70}")
print(f"DIAGNOSIS:")
print(f"{'='*70}")
print(f"\nIf Faster R-CNN scores are typically < 0.5, they will be filtered!")
print(f"\nSOLUTION: Lower confidence_threshold in visualization config:")
print(f"  visualization:")
print(f"    confidence_threshold: 0.05  # Instead of 0.5")
print(f"\nOr check why Faster R-CNN is producing low confidence scores.")
print(f"This could indicate:")
print(f"  1. Model needs more training")
print(f"  2. Model is uncertain about detections")
print(f"  3. Dataset labeling issues")
print(f"{'='*70}\n")

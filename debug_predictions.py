"""
Debug script to check Faster R-CNN prediction output format.
"""

from src.config.defaults import DEFAULTS
from src.models.registry import get_model
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


# Create a dummy Faster R-CNN model
config = {'models': DEFAULTS['models']}
model = get_model('faster_rcnn', num_classes=2, config=config)
model.eval()

# Create dummy input
dummy_images = [torch.randn(3, 640, 640)]

# Get predictions
with torch.no_grad():
    predictions = model(dummy_images)

# Print prediction structure
print("="*70)
print("FASTER R-CNN PREDICTION FORMAT")
print("="*70)
print(f"Type: {type(predictions)}")
print(f"Length: {len(predictions)}")

if predictions:
    print(f"\nFirst prediction type: {type(predictions[0])}")
    print(f"First prediction keys: {predictions[0].keys()}")

    for key in predictions[0].keys():
        value = predictions[0][key]
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        print(f"  Sample: {value[:3] if len(value) > 0 else 'empty'}")

print("\n" + "="*70)

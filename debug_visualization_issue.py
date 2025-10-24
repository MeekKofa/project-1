"""
Debug script to understand why Faster R-CNN isn't showing prediction scores in visualizations.
"""

import json
from pathlib import Path

# Check if prediction files exist
faster_rcnn_pred_dir = Path('outputs/cattle/faster_rcnn/predictions')
yolov8_pred_dir = Path('outputs/cattle/yolov8_resnet/predictions')

print("="*70)
print("DEBUGGING VISUALIZATION SCORE ISSUE")
print("="*70)


def check_predictions(pred_dir, model_name):
    print(f"\n{model_name} Predictions:")
    print("-" * 70)

    if not pred_dir.exists():
        print(f"  ❌ Directory not found: {pred_dir}")
        return

    # Find prediction JSON files
    pred_files = list(pred_dir.glob('*.json'))
    if not pred_files:
        print(f"  ❌ No prediction files found")
        return

    print(f"  ✓ Found {len(pred_files)} prediction file(s)")

    # Read first prediction file
    pred_file = pred_files[0]
    print(f"  Reading: {pred_file.name}")

    with open(pred_file) as f:
        data = json.load(f)

    # Check structure
    if 'predictions' in data:
        preds = data['predictions']
        print(f"\n  Total predictions: {len(preds)}")

        # Check first few predictions
        for img_id, pred_data in list(preds.items())[:3]:
            print(f"\n  Image {img_id}:")
            print(f"    Boxes: {len(pred_data.get('boxes', []))}")
            print(f"    Scores: {len(pred_data.get('scores', []))}")
            print(f"    Labels: {len(pred_data.get('labels', []))}")

            if pred_data.get('scores'):
                # First 5 scores
                print(f"    Score values: {pred_data['scores'][:5]}")
            else:
                print(f"    ⚠️  NO SCORES FOUND!")


# Check both models
check_predictions(faster_rcnn_pred_dir, "Faster R-CNN")
check_predictions(yolov8_pred_dir, "YOLOv8")

print("\n" + "="*70)

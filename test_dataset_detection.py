#!/usr/bin/env python3
"""
Test dataset detection to debug class loading issues.
This script tests WITHOUT importing torch-dependent modules.
"""

import json
from pathlib import Path

print("="*70)
print("DATASET DETECTION TEST (NO TORCH REQUIRED)")
print("="*70)

# Simulate what the detection code does
print("\n1. Checking dataset/cattle/data.yaml...")
yaml_path = Path('dataset/cattle/data.yaml')
if yaml_path.exists():
    print(f"   ✅ Found")
    try:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            print(f"      - nc: {data.get('nc', data.get('num_classes'))}")
            print(f"      - names: {data.get('names', [])}")
    except ImportError:
        print("      ⚠️  yaml module not installed, skipping content read")
else:
    print(f"   ❌ Not found (expected - cattle dataset is actually cattlebody)")

# Check file existence
print("\n3. Checking file paths...")
files_to_check = [
    'dataset/cattlebody/data.yaml',
    'processed_data/cattle/dataset_info.json',
    'processed_data/cattle/train',
    'processed_data/cattle/train/labels',
]

for filepath in files_to_check:
    path = Path(filepath)
    exists = "✅" if path.exists() else "❌"
    print(f"   {exists} {filepath}")

# Read the JSON directly
print("\n4. Reading dataset_info.json directly...")
json_path = Path('processed_data/cattle/dataset_info.json')
if json_path.exists():
    import json
    with open(json_path) as f:
        data = json.load(f)
        print(f"   ✅ File contents:")
        for key, value in data.items():
            print(f"      - {key}: {value}")
else:
    print(f"   ❌ File not found")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

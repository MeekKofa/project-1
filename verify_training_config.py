#!/usr/bin/env python3
"""
Quick verification that training configuration is correct.
Run this before attempting training.
"""

import json
from pathlib import Path


def check_dataset_info():
    """Check if dataset_info.json has required fields."""
    print("="*70)
    print("CHECKING DATASET CONFIGURATION")
    print("="*70)

    dataset_info_path = Path("processed_data/cattle/dataset_info.json")

    if not dataset_info_path.exists():
        print("❌ dataset_info.json not found!")
        return False

    with open(dataset_info_path) as f:
        info = json.load(f)

    required_fields = ['num_classes', 'class_names',
                       'num_train', 'num_val', 'num_test']
    missing = [field for field in required_fields if field not in info]

    if missing:
        print(f"❌ Missing fields: {missing}")
        return False

    print(f"✅ Dataset: {info['dataset_name']}")
    print(f"✅ Classes: {info['num_classes']}")
    print(f"✅ Class names: {info['class_names']}")
    print(
        f"✅ Train: {info['num_train']}, Val: {info['num_val']}, Test: {info['num_test']}")

    if info['num_classes'] is None or info['num_classes'] == 0:
        print("❌ num_classes is None or 0!")
        return False

    return True


def check_model_config():
    """Check if model config has correct parameters."""
    print("\n" + "="*70)
    print("CHECKING MODEL CONFIGURATION")
    print("="*70)

    try:
        from src.config.defaults import DEFAULTS

        yolov8_config = DEFAULTS.get('models', {}).get('yolov8', {})

        # Check for invalid parameters
        invalid_params = ['backbone', 'depths',
                          'widths', 'activation', 'use_sppf']
        found_invalid = [p for p in invalid_params if p in yolov8_config]

        if found_invalid:
            print(f"⚠️  Found deprecated parameters: {found_invalid}")
            print("   (These will be ignored but should be removed)")

        # Check for required parameters
        required = ['pretrained', 'in_channels', 'base_channels', 'config']
        missing = [p for p in required if p not in yolov8_config]

        if missing:
            print(f"❌ Missing required parameters: {missing}")
            return False

        print("✅ Model config has correct structure")
        print(f"   - pretrained: {yolov8_config.get('pretrained')}")
        print(f"   - in_channels: {yolov8_config.get('in_channels')}")
        print(f"   - base_channels: {yolov8_config.get('base_channels')}")
        print(f"   - config: {yolov8_config.get('config')}")

        return True

    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def check_model_registry():
    """Check if model registry has correct setup."""
    print("\n" + "="*70)
    print("CHECKING MODEL REGISTRY")
    print("="*70)

    try:
        from src.models.registry import MODEL_REGISTRY

        if 'yolov8_resnet' not in MODEL_REGISTRY:
            print("❌ yolov8_resnet not in registry!")
            return False

        model_info = MODEL_REGISTRY['yolov8_resnet']

        print("✅ yolov8_resnet found in registry")
        print(f"   - Module: {model_info.get('module')}")
        print(f"   - Class: {model_info.get('class_name')}")
        print(f"   - Init args: {model_info.get('init_args')}")
        print(f"   - Config key: {model_info.get('config_key')}")

        # Check init_args
        init_args = model_info.get('init_args', {})
        if 'backbone_type' not in init_args:
            print("❌ init_args missing backbone_type!")
            return False

        if init_args['backbone_type'] != 'resnet50':
            print(
                f"⚠️  backbone_type is {init_args['backbone_type']}, expected resnet50")

        return True

    except Exception as e:
        print(f"❌ Error checking registry: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION VERIFICATION")
    print("="*70 + "\n")

    checks = [
        ("Dataset Info", check_dataset_info),
        ("Model Config", check_model_config),
        ("Model Registry", check_model_registry),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    if all_passed:
        print("\n🎉 All checks passed! Ready to train.")
        print("\nRun training with:")
        print("  python train.py train -m yolov8_resnet -d cattle -e 2 -b 4")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")

    print("="*70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())

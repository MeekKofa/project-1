#!/usr/bin/env python3
"""
System Verification Script
Run this to verify all components are properly installed and wired.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def check_component(name, func):
    """Check a component and report status."""
    try:
        func()
        print(f"✅ {name}")
        return True
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False


def check_imports():
    """Check all required imports."""
    import torch
    import torchvision
    import yaml
    import numpy as np
    from PIL import Image


def check_cli():
    """Check CLI module."""
    from src.cli.args import parse_args, create_parser
    parser = create_parser()
    assert parser is not None


def check_config():
    """Check config module."""
    from src.config.manager import ConfigManager, get_config
    from src.config.defaults import DEFAULTS
    assert DEFAULTS is not None
    assert 'training' in DEFAULTS


def check_model_registry():
    """Check model registry."""
    from src.models.registry import get_model, get_model_info, list_models
    models = list_models()
    assert 'faster_rcnn' in models
    assert 'yolov8_resnet' in models
    assert 'yolov8_csp' in models


def check_data_registry():
    """Check data registry."""
    from src.loaders.registry import get_dataset, get_dataset_info, list_datasets
    datasets = list_datasets()
    assert 'cattle' in datasets


def check_training():
    """Check training modules."""
    from src.training.orchestrator import TrainingOrchestrator, train_model
    from src.training.base_trainer import BaseTrainer
    from src.training.checkpoints import CheckpointManager
    from src.training.loops.detection import DetectionTrainer


def check_loaders():
    """Check data loaders."""
    from src.loaders.base_loader import BaseDetectionLoader
    from src.loaders.cattle_loader import CattleDetectionDataset
    from src.loaders.transforms import (
        get_train_transforms,
        get_val_transforms,
        detection_collate_fn
    )


def check_main():
    """Check main entry point."""
    from train import main, handle_train, handle_eval, handle_preprocess


def main_verify():
    """Run all verifications."""
    print("\n" + "="*60)
    print("SYSTEM VERIFICATION")
    print("="*60 + "\n")

    checks = [
        ("Required Imports (torch, yaml, etc.)", check_imports),
        ("CLI Module (src/cli/args.py)", check_cli),
        ("Config Module (src/config/)", check_config),
        ("Model Registry (src/models/registry.py)", check_model_registry),
        ("Data Registry (src/loaders/registry.py)", check_data_registry),
        ("Training Modules (src/training/)", check_training),
        ("Data Loaders (src/loaders/)", check_loaders),
        ("Main Entry Point (train.py)", check_main),
    ]

    results = []
    for name, func in checks:
        results.append(check_component(name, func))

    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("="*60)
        print("\n🎉 System is ready to use!")
        print("\nTry running:")
        print("  python train.py train -m yolov8_resnet -d cattle -e 2 -b 4")
    else:
        print(f"⚠️  SOME CHECKS FAILED ({passed}/{total})")
        print("="*60)
        print("\nPlease fix the issues above before running training.")

    print()
    return passed == total


if __name__ == '__main__':
    success = main_verify()
    sys.exit(0 if success else 1)

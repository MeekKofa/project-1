"""
Data loading utilities.

Provides data loaders, transforms, and registry for detection tasks.
"""

from .base_loader import BaseDetectionLoader
from .cattle_loader import CattleDetectionDataset
from .registry import (
    get_dataset,
    get_dataset_info,
    list_datasets,
    register_dataset,
    DATASET_REGISTRY
)
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
    detection_collate_fn,
    Compose,
    ToTensor,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    ColorJitter,
    RandomCrop,
)

__all__ = [
    # Loaders
    'BaseDetectionLoader',
    'CattleDetectionDataset',

    # Registry
    'get_dataset',
    'get_dataset_info',
    'list_datasets',
    'register_dataset',
    'DATASET_REGISTRY',

    # Transforms
    'get_train_transforms',
    'get_val_transforms',
    'get_test_transforms',
    'detection_collate_fn',
    'Compose',
    'ToTensor',
    'Normalize',
    'Resize',
    'RandomHorizontalFlip',
    'ColorJitter',
    'RandomCrop',
]

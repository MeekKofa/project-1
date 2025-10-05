"""
Data Loader Registry - Central orchestrator for all data loaders.
Add a new data loader by adding ONE entry to DATASET_REGISTRY.
"""

import importlib
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import Dataset

from src.loaders.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)


# ===================================================================
# DATASET REGISTRY - Add new datasets here
# ===================================================================
DATASET_REGISTRY = {
    'cattle': {
        'module': 'src.loaders.cattle_loader',
        'class_name': 'CattleDetectionDataset',
        'num_classes': None,  # Auto-detect from data
        'class_names': None,  # Auto-detect from data
        'format': 'yolo',     # yolo, coco, or pascal
    },

    'cattlebody': {
        'module': 'src.loaders.cattle_loader',
        'class_name': 'CattleDetectionDataset',
        'num_classes': None,  # Auto-detect
        'class_names': None,  # Auto-detect
        'format': 'yolo',
    },

    'cattleface': {
        'module': 'src.loaders.cattle_loader',
        'class_name': 'CattleDetectionDataset',
        'num_classes': None,  # Auto-detect
        'class_names': None,  # Auto-detect
        'format': 'yolo',
    },
}


# ===================================================================
# Auto-Detection Functions
# ===================================================================

def detect_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Auto-detect dataset information from files.
    Priority: data.yaml > dataset_info.json > directory scan

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary with num_classes, class_names, splits, etc.
    """
    info = {}

    # Try data.yaml (YOLO format)
    yaml_path = Path(f'dataset/{dataset_name}/data.yaml')
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        num_classes = data.get('nc', data.get('num_classes'))
        class_names = data.get('names') or data.get('class_names')

        if isinstance(class_names, dict):
            class_names = list(class_names.values())

        if class_names is None:
            class_names = []

        if num_classes is not None:
            info['num_classes'] = int(num_classes)
            info['class_names'] = class_names
            info['train_path'] = data.get(
                'train', f'dataset/{dataset_name}/train')
            info['val_path'] = data.get('val', f'dataset/{dataset_name}/val')
            info['test_path'] = data.get(
                'test', f'dataset/{dataset_name}/test')
            info['format'] = data.get('format', 'yolo')
            print(f"✓ Loaded dataset info from {yaml_path}")
            return info
        else:
            print(
                f"⚠ data.yaml found at {yaml_path} but missing 'nc'/'num_classes'. "
                "Falling back to other sources..."
            )

    # Try dataset_info.json (processed data)
    json_path = Path(f'processed_data/{dataset_name}/dataset_info.json')
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f) or {}

        num_classes = data.get('num_classes')
        class_names = data.get('class_names')

        if isinstance(class_names, dict):
            class_names = list(class_names.values())

        if class_names is None:
            class_names = []

        info['num_train'] = data.get('num_train')
        info['num_val'] = data.get('num_val')
        info['num_test'] = data.get('num_test')
        info['format'] = data.get('format', 'yolo')

        if num_classes is not None:
            info['num_classes'] = int(num_classes)
            info['class_names'] = class_names
            print(f"✓ Loaded dataset info from {json_path}")
            return info
        else:
            print(
                f"⚠ dataset_info.json found at {json_path} but missing 'num_classes'. "
                "Attempting directory scan..."
            )

    # Fallback: scan directories
    print(f"⚠ No data.yaml or dataset_info.json found, scanning directories...")
    info = _scan_dataset_dirs(dataset_name)

    return info


def _scan_dataset_dirs(dataset_name: str) -> Dict[str, Any]:
    """
    Scan dataset directories to detect info.

    Args:
        dataset_name: Name of dataset

    Returns:
        Detected dataset information
    """
    info = {
        'num_classes': None,
        'class_names': [],
        'format': 'yolo',
    }

    # Check different possible locations
    possible_paths = [
        Path(f'dataset/{dataset_name}'),
        Path(f'processed_data/{dataset_name}'),
    ]

    for base_path in possible_paths:
        if not base_path.exists():
            continue

        # Check for train directory
        train_path = base_path / 'train'
        if train_path.exists():
            # YOLO format: labels and images directories
            labels_path = train_path / 'labels'
            if labels_path.exists():
                info['format'] = 'yolo'
                # Count unique classes from label files
                class_ids = set()
                for label_file in labels_path.glob('*.txt'):
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))

                info['num_classes'] = len(class_ids)
                info['class_names'] = [f'class_{i}' for i in sorted(class_ids)]
                print(f"✓ Detected {info['num_classes']} classes from labels")
                return info

        break

    print(f"⚠ Could not auto-detect dataset info for '{dataset_name}'")
    print(f"  Please create data.yaml or dataset_info.json")

    return info


# ===================================================================
# Utility helpers
# ===================================================================

def _infer_image_size(config: Dict[str, Any]) -> int:
    """Infer target image size from configuration."""
    preprocess_cfg = config.get('preprocess', {})
    target_size = preprocess_cfg.get('target_size')

    if isinstance(target_size, (list, tuple)) and target_size:
        return int(target_size[0])

    if isinstance(target_size, int):
        return int(target_size)

    data_cfg = config.get('data', {})
    return int(data_cfg.get('img_size', 640))


def _find_dataset_root(dataset_name: str, info: Dict[str, Any], split: str) -> Optional[Path]:
    """Determine the most likely dataset root directory."""
    candidates: List[Path] = []

    for key in ('train_path', 'val_path', 'test_path'):
        path = info.get(key)
        if path:
            p = Path(path)
            candidates.extend([p, p.parent, p.parent.parent])

    candidates.extend([
        Path(f'processed_data/{dataset_name}'),
        Path(f'dataset/{dataset_name}'),
    ])

    visited: List[Path] = []
    for candidate in candidates:
        if not isinstance(candidate, Path):
            continue

        if candidate in visited:
            continue
        visited.append(candidate)

        if not candidate.exists():
            continue

        potential_roots = [candidate]
        if candidate.name in {'images', 'labels', split}:
            potential_roots.append(candidate.parent)
            potential_roots.append(candidate.parent.parent)
        else:
            potential_roots.append(candidate.parent)

        for root in potential_roots:
            if root is None or not isinstance(root, Path):
                continue
            if not root.exists():
                continue

            if (root / split / 'images').exists():
                return root
            if (root / 'images' / split).exists():
                return root
            if (root / split).exists() and (root / split / 'images').exists():
                return root

    return None


# ===================================================================
# Registry Functions
# ===================================================================

def get_dataset(
    dataset_name: str,
    split: str,
    config: Dict[str, Any]
) -> Dataset:
    """
    Load dataset from registry.

    Args:
        dataset_name: Name of dataset in registry
        split: 'train', 'val', or 'test'
        config: Full configuration dictionary

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset not found in registry
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ', '.join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry. "
            f"Available datasets: {available}"
        )

    info = DATASET_REGISTRY[dataset_name]

    # Import module dynamically
    try:
        module = importlib.import_module(info['module'])
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{info['module']}' for dataset '{dataset_name}': {e}"
        )

    # Get dataset class
    if not hasattr(module, info['class_name']):
        raise AttributeError(
            f"Module '{info['module']}' has no class '{info['class_name']}'"
        )

    dataset_class = getattr(module, info['class_name'])

    # Get dataset-specific config
    dataset_config = config.get('datasets', {}).get(dataset_name, {})

    # Merge with global data config
    data_config = config.get('data', {})
    dataset_kwargs = {**data_config, **dataset_config}

    # Determine root directory
    root_dir = dataset_kwargs.get('root_dir')
    if not root_dir:
        inferred_root = _find_dataset_root(dataset_name, info, split)
        if inferred_root is None:
            raise FileNotFoundError(
                f"Could not determine root directory for dataset '{dataset_name}'. "
                "Provide 'root_dir' in the config or ensure the dataset exists in processed_data/."
            )
        dataset_kwargs['root_dir'] = str(inferred_root)

    # Determine image size & augmentation
    image_size = dataset_kwargs.get('image_size')
    if image_size is None:
        image_size = _infer_image_size(config)
        dataset_kwargs['image_size'] = image_size

    if isinstance(image_size, (list, tuple)):
        dataset_kwargs['image_size'] = int(image_size[0])

    # Augmentation enabled only for training split
    augment_flag = config.get('augmentation', {}).get('enabled', True) and split == 'train'
    dataset_kwargs.setdefault('augment', augment_flag)

    # Attach transforms if not provided
    if 'transform' not in dataset_kwargs or dataset_kwargs['transform'] is None:
        size_int = int(dataset_kwargs['image_size'])
        if split == 'train' and dataset_kwargs['augment']:
            dataset_kwargs['transform'] = get_train_transforms(size_int)
        elif split == 'val':
            dataset_kwargs['transform'] = get_val_transforms(size_int)
        else:
            dataset_kwargs['transform'] = get_test_transforms(size_int)

    # Ensure format and metadata
    dataset_kwargs.setdefault('format', info.get('format', info.get('dataset_format', 'yolo')))

    if info.get('class_names'):
        dataset_kwargs.setdefault('class_names', info['class_names'])
    if info.get('num_classes') is not None:
        dataset_kwargs.setdefault('num_classes', info['num_classes'])

    # Initialize dataset
    try:
        dataset = dataset_class(
            dataset_name=dataset_name,
            split=split,
            **dataset_kwargs
        )
    except Exception as e:
        raise RuntimeError(
            f"Error initializing dataset '{dataset_name}' split '{split}': {e}"
        )

    print(f"✓ Loaded dataset: {dataset_name}/{split}")
    print(f"  - Samples: {len(dataset)}")

    return dataset


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a specific dataset (with auto-detection).

    Args:
        dataset_name: Name of dataset in registry

    Returns:
        Dataset information dictionary

    Raises:
        ValueError: If dataset not found
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry")

    info = DATASET_REGISTRY[dataset_name].copy()

    # Auto-detect missing information
    if info['num_classes'] is None or info['class_names'] is None:
        detected = detect_dataset_info(dataset_name)
        info.update(detected)

    return info


def list_datasets() -> Dict[str, str]:
    """
    List all available datasets.

    Returns:
        Dictionary mapping dataset names to formats
    """
    return {name: info['format'] for name, info in DATASET_REGISTRY.items()}


def register_dataset(
    name: str,
    module: str,
    class_name: str,
    format: str = 'yolo',
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None
):
    """
    Register a new dataset at runtime.

    Args:
        name: Unique dataset name
        module: Python module path
        class_name: Class name in module
        format: Dataset format ('yolo', 'coco', or 'pascal')
        num_classes: Number of classes (None for auto-detect)
        class_names: List of class names (None for auto-detect)
    """
    if name in DATASET_REGISTRY:
        print(f"Warning: Overwriting existing dataset '{name}'")

    DATASET_REGISTRY[name] = {
        'module': module,
        'class_name': class_name,
        'num_classes': num_classes,
        'class_names': class_names,
        'format': format,
    }

    print(f"✓ Registered dataset: {name}")


# ===================================================================
# Helper Functions
# ===================================================================

def print_registry():
    """Print all registered datasets."""
    print("\n" + "=" * 60)
    print("AVAILABLE DATASETS")
    print("=" * 60)

    for name, info in DATASET_REGISTRY.items():
        print(f"\n{name}:")
        print(f"  Module: {info['module']}")
        print(f"  Class: {info['class_name']}")
        print(f"  Format: {info['format']}")

        # Try to get auto-detected info
        try:
            full_info = get_dataset_info(name)
            if full_info.get('num_classes'):
                print(f"  Classes: {full_info['num_classes']}")
            if full_info.get('class_names'):
                print(f"  Names: {full_info['class_names']}")
        except Exception:
            pass

    print("\n" + "=" * 60)


if __name__ == '__main__':
    # Test registry
    print_registry()

    # Test auto-detection
    print("\n" + "=" * 60)
    print("AUTO-DETECTION TEST")
    print("=" * 60)

    for dataset_name in ['cattle', 'cattlebody', 'cattleface']:
        print(f"\n{dataset_name}:")
        try:
            info = get_dataset_info(dataset_name)
            print(f"  ✓ Num classes: {info.get('num_classes', 'Unknown')}")
            print(f"  ✓ Class names: {info.get('class_names', 'Unknown')}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

"""
Model Registry - Central orchestrator for all models.
Add a new model by adding ONE entry to MODEL_REGISTRY.
"""

import importlib
from typing import Dict, Any, Optional
import torch.nn as nn


# ===================================================================
# MODEL REGISTRY - Add new models here
# ===================================================================
MODEL_REGISTRY = {
    'faster_rcnn': {
        'module': 'src.models.faster_rcnn.model',
        'class_name': 'FasterRCNNModel',
        'config_key': 'faster_rcnn',
        'description': 'Faster R-CNN with light backbone',
    },

    'yolov8_resnet': {
        'module': 'src.models.yolov8.model',
        'class_name': 'YOLOv8Model',
        'init_args': {'backbone_type': 'resnet50'},
        'config_key': 'yolov8',
        'description': 'YOLOv8 with ResNet-50 backbone',
    },

    'yolov8_csp': {
        'module': 'src.models.yolov8.model',
        'class_name': 'YOLOv8Model',
        'init_args': {'backbone_type': 'csp'},
        'config_key': 'yolov8',
        'description': 'YOLOv8 with CSP backbone (default)',
    },
}


# ===================================================================
# Registry Functions
# ===================================================================

def get_model(
    model_name: str,
    num_classes: int,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Load model from registry.

    Args:
        model_name: Name of model in registry
        num_classes: Number of output classes (auto-detected from dataset)
        config: Full configuration dictionary

    Returns:
        Initialized model

    Raises:
        ValueError: If model not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {available}"
        )

    info = MODEL_REGISTRY[model_name]

    # Import module dynamically
    try:
        module = importlib.import_module(info['module'])
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{info['module']}' for model '{model_name}': {e}"
        )

    # Get model class
    if not hasattr(module, info['class_name']):
        raise AttributeError(
            f"Module '{info['module']}' has no class '{info['class_name']}'"
        )

    model_class = getattr(module, info['class_name'])

    # Get model-specific config
    config_key = info.get('config_key', model_name)
    model_config = config.get('models', {}).get(config_key, {})

    # Get initialization arguments
    init_args = info.get('init_args', {})

    # Merge all args (init_args override config to allow model variants)
    model_kwargs = {
        'num_classes': num_classes,
        **model_config,
        **init_args  # init_args take precedence over config
    }

    # Initialize model
    try:
        model = model_class(**model_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Error initializing model '{model_name}' with args {model_kwargs}: {e}"
        )

    print(f"✓ Loaded model: {model_name} ({info['description']})")
    print(f"  - Classes: {num_classes}")
    print(f"  - Config: {model_kwargs}")

    return model


def list_models() -> Dict[str, str]:
    """
    List all available models.

    Returns:
        Dictionary mapping model names to descriptions
    """
    return {name: info['description'] for name, info in MODEL_REGISTRY.items()}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.

    Args:
        model_name: Name of model in registry

    Returns:
        Model information dictionary

    Raises:
        ValueError: If model not found
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")

    return MODEL_REGISTRY[model_name].copy()


def register_model(
    name: str,
    module: str,
    class_name: str,
    config_key: str,
    description: str,
    init_args: Optional[Dict[str, Any]] = None
):
    """
    Register a new model at runtime.

    Args:
        name: Unique model name
        module: Python module path (e.g., 'src.models.retinanet')
        class_name: Class name in module
        config_key: Key in config['models'] for this model
        description: Human-readable description
        init_args: Optional default initialization arguments
    """
    if name in MODEL_REGISTRY:
        print(f"Warning: Overwriting existing model '{name}'")

    MODEL_REGISTRY[name] = {
        'module': module,
        'class_name': class_name,
        'config_key': config_key,
        'description': description,
    }

    if init_args:
        MODEL_REGISTRY[name]['init_args'] = init_args

    print(f"✓ Registered model: {name}")


# ===================================================================
# Helper Functions
# ===================================================================

def print_registry():
    """Print all registered models."""
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)

    for name, info in MODEL_REGISTRY.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Module: {info['module']}")
        print(f"  Class: {info['class_name']}")
        if 'init_args' in info:
            print(f"  Init Args: {info['init_args']}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    # Test registry
    print_registry()

    # Test getting model info
    print("\nModel info for 'yolov8_resnet':")
    info = get_model_info('yolov8_resnet')
    print(info)

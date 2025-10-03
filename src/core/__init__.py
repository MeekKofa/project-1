"""
Core abstractions and design patterns for the detection system.

This module provides base classes and registries for:
- Models
- Trainers
- Datasets
- Evaluators
"""

from .model_base import DetectionModelBase
from .trainer_base import TrainerBase
from .registry import ModelRegistry, DatasetRegistry

__all__ = [
    'DetectionModelBase',
    'TrainerBase',
    'ModelRegistry',
    'DatasetRegistry',
]

"""
Base Data Loader - Abstract base class for all data loaders.

Provides common functionality for detection data loaders.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseDetectionLoader(Dataset, ABC):
    """
    Abstract base class for detection data loaders.

    All detection loaders should inherit from this class.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        image_size: int = 640,
        augment: bool = False,
        **_: Any,
    ):
        """
        Initialize base detection loader.

        Args:
            root_dir: Root directory containing the data
            split: Data split ('train', 'val', 'test')
            transform: Image transformations
            target_transform: Target transformations
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.augment = augment

        # Validate split
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(
                f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

        # Placeholder for samples - subclasses should populate
        self.samples = []

    @abstractmethod
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load annotations from disk.

        Returns:
            List of sample dictionaries with 'image_path', 'boxes', 'labels', etc.
        """
        pass

    @abstractmethod
    def _load_sample(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a single sample (image and target).

        Args:
            idx: Sample index

        Returns:
            image: PIL Image or tensor
            target: Dictionary with 'boxes', 'labels', etc.
        """
        pass

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a sample.

        Args:
            idx: Sample index

        Returns:
            image: Tensor [C, H, W]
            target: Dictionary with 'boxes', 'labels', etc.
        """
        # Load sample
        image, target = self._load_sample(idx)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a sample without loading it.

        Args:
            idx: Sample index

        Returns:
            Sample information dictionary
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range [0, {len(self.samples)})")

        return self.samples[idx].copy()

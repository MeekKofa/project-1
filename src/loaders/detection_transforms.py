"""
Transforms for object detection tasks.
These transforms handle both image and target transformations.
"""

from typing import Tuple, Dict, Any
import torch
from torchvision import transforms as T

class DetectionTransform:
    """Base class for detection transforms that handle both image and target."""
    def __call__(self, image, target):
        raise NotImplementedError

class ToTensor(DetectionTransform):
    """Convert image to tensor, leaving target unchanged."""
    def __call__(self, image, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Convert image to tensor
        to_tensor = T.ToTensor()
        image = to_tensor(image)
        return image, target

class Compose(DetectionTransform):
    """Compose multiple detection transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
"""
Data transforms for detection tasks.

Provides augmentation and preprocessing transforms.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import random
from typing import Tuple, Dict, Any, Optional
import numpy as np


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image to tensor."""

    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image with mean and std."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    """Resize image to target size."""

    def __init__(self, size: int = 640):
        self.size = size

    def __call__(self, image, target=None):
        # Original size
        orig_w, orig_h = image.size

        # Resize image
        image = F.resize(image, [self.size, self.size])

        # Update boxes if target provided
        if target is not None and 'boxes' in target:
            boxes = target['boxes']
            # Scale boxes to new size
            scale_x = self.size / orig_w
            scale_y = self.size / orig_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes

        return image, target


class RandomHorizontalFlip:
    """Randomly flip image horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            # Flip image
            image = F.hflip(image)

            # Flip boxes
            if target is not None and 'boxes' in target:
                width = image.width
                boxes = target['boxes']
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes

        return image, target


class ColorJitter:
    """Randomly change brightness, contrast, saturation."""

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1
    ):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image, target=None):
        image = self.transform(image)
        return image, target


class RandomCrop:
    """Randomly crop image and adjust boxes."""

    def __init__(self, min_scale: float = 0.5):
        self.min_scale = min_scale

    def __call__(self, image, target=None):
        if target is None or 'boxes' not in target:
            return image, target

        width, height = image.size
        boxes = target['boxes']

        # Random crop parameters
        scale = random.uniform(self.min_scale, 1.0)
        new_width = int(width * scale)
        new_height = int(height * scale)

        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)

        # Crop image
        image = F.crop(image, top, left, new_height, new_width)

        # Adjust boxes
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left).clamp(0, new_width)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top).clamp(0, new_height)

        # Filter boxes with very small area
        keep = ((boxes[:, 2] - boxes[:, 0]) >
                1) & ((boxes[:, 3] - boxes[:, 1]) > 1)

        if keep.sum() == 0:
            # No valid boxes, return original
            return image, target

        target['boxes'] = boxes[keep]
        target['labels'] = target['labels'][keep]

        return image, target


def get_train_transforms(img_size: int = 640) -> Compose:
    """
    Get training transforms with augmentation.

    Args:
        img_size: Target image size

    Returns:
        Composed transforms
    """
    return Compose([
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        RandomHorizontalFlip(p=0.5),
        Resize(img_size),
        ToTensor(),
        Normalize(),
    ])


def get_val_transforms(img_size: int = 640) -> Compose:
    """
    Get validation transforms (no augmentation).

    Args:
        img_size: Target image size

    Returns:
        Composed transforms
    """
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(),
    ])


def get_test_transforms(img_size: int = 640) -> Compose:
    """
    Get test transforms (no augmentation).

    Args:
        img_size: Target image size

    Returns:
        Composed transforms
    """
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(),
    ])


# Collate function for DataLoader
def detection_collate_fn(batch):
    """
    Collate function for detection batches.

    Args:
        batch: List of (image, target) tuples

    Returns:
        images: Batched images [B, C, H, W]
        targets: List of target dictionaries
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # Stack images
    images = torch.stack(images, dim=0)

    return images, targets

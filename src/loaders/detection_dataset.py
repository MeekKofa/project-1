"""
Universal Detection Dataset.

Generic dataset that works with all detection models.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as T
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class DetectionDataset(Dataset):
    """
    Universal detection dataset for object detection tasks.

    Supports YOLO format (txt annotations) and can be extended for other formats.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_size: int = 640,
        augment: bool = False,
        normalize: bool = True,
        transforms: Optional[Callable] = None
    ):
        """
        Initialize detection dataset.

        Args:
            images_dir: Path to directory containing images
            labels_dir: Path to directory containing label files
            image_size: Target image size (will resize to square)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images to [0, 1]
            transforms: Optional custom transforms
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.custom_transforms = transforms
        self.hflip_prob = 0.5 if augment else 0.0

        # Find all images
        self.image_files = self._find_images()

        # Build transforms
        self.transforms = self._build_transforms()

        logger.info(f"Loaded {len(self.image_files)} images from {images_dir}")

    def _find_images(self) -> List[str]:
        """Find all image files in the directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []

        for filename in sorted(os.listdir(self.images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(filename)

        return image_files

    def _build_transforms(self) -> Callable:
        """Build image transforms."""
        transform_list = []

        # Resize to target size
        transform_list.append(T.Resize((self.image_size, self.image_size)))

        # Data augmentation (only during training)
        if self.augment:
            transform_list.extend([
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
            ])

        # Convert to tensor
        transform_list.append(T.ToTensor())

        # Normalize if requested
        if self.normalize:
            # ImageNet normalization
            transform_list.append(
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            )

        return T.Compose(transform_list)

    def _load_yolo_annotation(self, label_path: str, img_width: int, img_height: int) -> Dict[str, torch.Tensor]:
        """
        Load YOLO format annotation.

        YOLO format: class cx cy w h (normalized to [0, 1])

        Args:
            label_path: Path to label file
            img_width: Original image width
            img_height: Original image height

        Returns:
            Dict with 'boxes' and 'labels'
        """
        boxes = []
        labels = []

        if not os.path.exists(label_path):
            # No annotations for this image
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            }

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # Convert from normalized cxcywh to pixel xyxy
                    # First to pixel coordinates
                    cx_px = cx * img_width
                    cy_px = cy * img_height
                    w_px = w * img_width
                    h_px = h * img_height

                    # Then to xyxy
                    x1 = cx_px - w_px / 2
                    y1 = cy_px - h_px / 2
                    x2 = cx_px + w_px / 2
                    y2 = cy_px + h_px / 2

                    # Clip to image bounds
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))

                    # Only keep valid boxes
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Skipping invalid line in {label_path}: {line.strip()}")
                    continue

        # Convert to tensors
        if len(boxes) == 0:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            }

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return {'boxes': boxes, 'labels': labels}

    def _scale_boxes(self, boxes: torch.Tensor, orig_width: int, orig_height: int) -> torch.Tensor:
        """
        Scale boxes from original image size to target size.

        Args:
            boxes: [N, 4] boxes in xyxy format (original scale)
            orig_width: Original image width
            orig_height: Original image height

        Returns:
            [N, 4] boxes scaled to target size
        """
        if boxes.numel() == 0:
            return boxes

        # Scale factors
        scale_x = self.image_size / orig_width
        scale_y = self.image_size / orig_height

        # Scale boxes
        boxes[:, [0, 2]] *= scale_x  # x1, x2
        boxes[:, [1, 3]] *= scale_y  # y1, y2

        # Clip to image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, self.image_size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, self.image_size)

        return boxes

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get image and target.

        Args:
            idx: Index

        Returns:
            Tuple of (image, target)
            - image: [3, H, W] tensor
            - target: Dict with 'boxes' [N, 4] and 'labels' [N]
        """
        # Load image
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)

        try:
            image = Image.open(img_path).convert('RGB')
            orig_width, orig_height = image.size
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return dummy data
            image = Image.new('RGB', (self.image_size, self.image_size))
            orig_width, orig_height = self.image_size, self.image_size

        # Load annotations
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_filename)
        target = self._load_yolo_annotation(
            label_path, orig_width, orig_height)
        
        # Add image metadata
        target['image_id'] = torch.tensor([idx], dtype=torch.int64)  # Use index as image ID
        target['image_path'] = img_path
        target['original_size'] = torch.tensor([orig_height, orig_width], dtype=torch.int64)

        # Geometric augmentations that require box updates
        if self.augment and self.hflip_prob > 0:
            if torch.rand(1).item() < self.hflip_prob:
                image = ImageOps.mirror(image)
                boxes = target['boxes'].clone()
                if boxes.numel() > 0:
                    x1 = boxes[:, 0].clone()
                    x2 = boxes[:, 2].clone()
                    boxes[:, 0] = orig_width - x2
                    boxes[:, 2] = orig_width - x1
                    target['boxes'] = boxes

        # Apply image transforms
        if self.custom_transforms is not None:
            image = self.custom_transforms(image)
        else:
            image = self.transforms(image)

        # Scale boxes to match resized image
        target['boxes'] = self._scale_boxes(
            target['boxes'], orig_width, orig_height)

        return image, target

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Custom collate function for batching.

        Args:
            batch: List of (image, target) tuples

        Returns:
            Tuple of (images, targets)
            - images: [B, 3, H, W] batched images
            - targets: List of B target dicts
        """
        images = []
        targets = []

        for image, target in batch:
            images.append(image)
            targets.append(target)

        # Stack images
        images = torch.stack(images, dim=0)

        return images, targets


def create_detection_dataloaders(
    train_images_dir: str,
    train_labels_dir: str,
    val_images_dir: str,
    val_labels_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 640
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        train_images_dir: Training images directory
        train_labels_dir: Training labels directory
        val_images_dir: Validation images directory
        val_labels_dir: Validation labels directory
        batch_size: Batch size
        num_workers: Number of dataloader workers
        image_size: Target image size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = DetectionDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        image_size=image_size,
        augment=True,
        normalize=True
    )

    val_dataset = DetectionDataset(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        image_size=image_size,
        augment=False,
        normalize=True
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=DetectionDataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=DetectionDataset.collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader

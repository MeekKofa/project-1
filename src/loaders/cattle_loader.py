"""
Cattle Detection Data Loader.

Loads cattle detection data in YOLO format.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import yaml
import json

from .base_loader import BaseDetectionLoader

logger = logging.getLogger(__name__)


class CattleDetectionDataset(BaseDetectionLoader):
    """
    Cattle Detection Data Loader.

    Supports YOLO format:
    - YOLO: images/ and labels/ directories with .txt files
    
    Note: COCO and Pascal VOC formats are not currently supported.
    Please convert your data to YOLO format first.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 640,
        format: str = 'yolo',
        transform: Optional[Any] = None,
        augment: bool = False,
        class_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize cattle detection dataset.

        Args:
            root_dir: Root directory containing dataset
            split: Dataset split ('train', 'val', or 'test')
            image_size: Target image size for resizing
            format: Annotation format ('yolo', 'coco', or 'pascal')
            transform: Optional transforms to apply
            augment: Whether to apply data augmentation
        """
        super().__init__(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            transform=transform,
            augment=augment,
            **kwargs,
        )

        self.format = format.lower()
        self.root_path = Path(root_dir)

        # Resolve class metadata
        if class_names is not None:
            self.class_names = list(class_names)
        else:
            self.class_names = self._load_class_names()

        if num_classes is not None:
            self.num_classes = int(num_classes)
        else:
            self.num_classes = len(self.class_names)

        if len(self.class_names) != self.num_classes:
            logger.warning(
                "Mismatch between class_names (%d) and num_classes (%d). Adjusting to match.",
                len(self.class_names),
                self.num_classes,
            )
            if len(self.class_names) < self.num_classes:
                start_idx = len(self.class_names)
                self.class_names.extend(
                    [f'class_{i}' for i in range(start_idx, self.num_classes)]
                )
            else:
                self.class_names = self.class_names[:self.num_classes]

        # Validate format
        if self.format not in ['yolo', 'coco', 'pascal']:
            raise ValueError(
                f"Format '{self.format}' not supported. "
                f"Use 'yolo', 'coco', or 'pascal'"
            )

        # Only YOLO is fully implemented
        if self.format != 'yolo':
            raise NotImplementedError(
                f"{self.format.upper()} format is not yet implemented. "
                f"Please convert your dataset to YOLO format. "
                f"Expected structure: {{images/, labels/}} with .txt annotations"
            )

        # Load dataset samples
        self.samples = self._load_annotations()

        logger.info(
            f"Loaded {len(self.samples)} samples from {split} split "
            f"({self.num_classes} classes)"
        )

    def _load_class_names(self) -> List[str]:
        """Load class names from data.yaml or classes.txt."""
        # Try data.yaml first
        yaml_path = self.root_path / 'data.yaml'
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    names = data['names']
                    # Handle both dict and list formats
                    if isinstance(names, dict):
                        return [names[i] for i in sorted(names.keys())]
                    return names

        # Try classes.txt
        classes_path = self.root_path / 'classes.txt'
        if classes_path.exists():
            with open(classes_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]

        # Default to single class
        logger.warning(
            "No class names found in data.yaml or classes.txt. "
            "Using default: ['cattle']"
        )
        return ['cattle']

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load annotations based on format."""
        if self.format == 'yolo':
            return self._load_yolo_annotations()
        elif self.format == 'coco':
            return self._load_coco_annotations()
        elif self.format == 'pascal':
            return self._load_pascal_annotations()
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _load_yolo_annotations(self) -> List[Dict[str, Any]]:
        """Load YOLO format annotations (txt files)."""
        samples = []

        # Resolve image/label directories for various expected layouts
        candidate_pairs = [
            (self.root_path / self.split / 'images', self.root_path / self.split / 'labels'),
            (self.root_path / self.split, self.root_path / self.split / 'labels'),
            (self.root_path / 'images' / self.split, self.root_path / 'labels' / self.split),
            (self.root_path / 'images', self.root_path / 'labels'),
            (self.root_path, self.root_path / 'labels'),
        ]

        images_dir: Optional[Path] = None
        labels_dir: Optional[Path] = None

        for img_dir, lbl_dir in candidate_pairs:
            if img_dir.exists():
                images_dir = img_dir
                labels_dir = lbl_dir
                break

        if images_dir is None:
            raise FileNotFoundError(
                "Could not determine image directory for dataset. Checked: "
                + ", ".join(str(p[0]) for p in candidate_pairs)
            )

        if labels_dir is None:
            labels_dir = images_dir.parent / 'labels'

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))

        logger.info(f"Found {len(image_files)} images in {images_dir}")

        for img_path in sorted(image_files):
            # Get corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"

            # Load annotations from label file
            boxes = []
            labels = []

            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        # YOLO format: class_id x_center y_center width height
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Convert to [x_min, y_min, x_max, y_max] format
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2
                        y_max = y_center + height / 2

                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)

            # Create sample dict
            sample = {
                'image_path': str(img_path),
                'boxes': boxes,
                'labels': labels,
                'image_id': len(samples)
            }

            samples.append(sample)

        return samples

    def _load_coco_annotations(self) -> List[Dict[str, Any]]:
        """
        Load COCO format annotations (JSON).
        
        Note: COCO format is not currently supported.
        To use COCO format data:
        1. Convert COCO to YOLO format using roboflow or custom script
        2. Use the YOLO format loader (default)
        """
        raise NotImplementedError(
            "COCO format not supported. Please convert to YOLO format. "
            "Expected structure: {images/, labels/} with .txt annotations"
        )

    def _load_pascal_annotations(self) -> List[Dict[str, Any]]:
        """
        Load Pascal VOC format annotations (XML).
        
        Note: Pascal VOC format is not currently supported.
        To use Pascal VOC data:
        1. Convert Pascal VOC to YOLO format using conversion script
        2. Use the YOLO format loader (default)
        """
        raise NotImplementedError(
            "Pascal VOC format not supported. Please convert to YOLO format. "
            "Expected structure: {images/, labels/} with .txt annotations"
        )

    def _load_sample(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Load a single sample (image and target).

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, target) where target contains boxes and labels
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        img_width, img_height = image.size

        # Prepare target
        boxes = sample['boxes']
        labels = sample['labels']

        # Convert normalized coordinates to absolute
        if boxes:
            boxes_abs = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                boxes_abs.append([
                    x_min * img_width,
                    y_min * img_height,
                    x_max * img_width,
                    y_max * img_height
                ])
            boxes = torch.tensor(boxes_abs, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([sample['image_id']]),
            'orig_size': torch.tensor([img_height, img_width])
        }

        return image, target

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, target_dict)
        """
        image, target = self._load_sample(idx)

        # Guarantee targets are never None or invalid
        if target is None or not isinstance(target, dict):
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty(0, dtype=torch.int64)
            }
        else:
            if "boxes" not in target or not isinstance(target["boxes"], torch.Tensor):
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            if "labels" not in target or not isinstance(target["labels"], torch.Tensor):
                target["labels"] = torch.empty(0, dtype=torch.int64)

        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

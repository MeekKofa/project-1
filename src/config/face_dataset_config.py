#!/usr/bin/env python3
"""
Face Dataset Configuration
Special configuration for parallel image/annotation directory structure
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class FaceDatasetConfig:
    """Configuration for face datasets with parallel image/annotation directories"""

    def __init__(self, dataset_root: str, image_dir: str, annotation_dir: str):
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / image_dir
        self.annotation_dir = self.dataset_root / annotation_dir
        self._config = None
        self._analyze_dataset()

    def _analyze_dataset(self):
        """Analyze face dataset structure and classes"""
        config = {
            "dataset_root": str(self.dataset_root),
            "image_dir": str(self.image_dir),
            "annotation_dir": str(self.annotation_dir),
            "classes": {},
            "total_annotations": 0,
            "total_images": 0
        }

        # Count images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for ext in image_extensions:
            config["total_images"] += len(list(self.image_dir.glob(f"*{ext}")))
            config["total_images"] += len(list(self.image_dir.glob(f"*{ext.upper()}")))

        # Analyze annotations
        for label_file in self.annotation_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(float(parts[0]))
                            config["classes"][class_id] = config["classes"].get(class_id, 0) + 1
                            config["total_annotations"] += 1
            except Exception as e:
                logger.warning(f"Error reading {label_file}: {e}")

        # Calculate num_classes for model
        if config["classes"]:
            max_yolo_class = max(config["classes"].keys())
            config["num_classes"] = max_yolo_class + 2  # background + max_class + 1
            config["class_range"] = [min(config["classes"].keys()), max_yolo_class]
        else:
            config["num_classes"] = 2  # Default: background + 1 class
            config["class_range"] = [0, 0]

        self._config = config
        logger.info(f"Dataset analysis complete: {config['total_images']} images, {config['num_classes']} classes")

    @property
    def num_classes(self) -> int:
        """Get number of classes for model (including background)"""
        return self._config["num_classes"]

    @property
    def class_names(self) -> List[str]:
        """Get class names (if available)"""
        # Default names - using just numeric IDs
        num_classes = self.num_classes
        return ["background"] + [str(i) for i in range(1, num_classes)]

    def get_directories(self) -> Dict[str, str]:
        """Get image and annotation directories"""
        return {
            "images": str(self.image_dir),
            "annotations": str(self.annotation_dir)
        }

    def validate_dataset(self) -> List[str]:
        """Validate dataset configuration"""
        issues = []

        if not self.image_dir.exists():
            issues.append(f"Image directory not found: {self.image_dir}")
        if not self.annotation_dir.exists():
            issues.append(f"Annotation directory not found: {self.annotation_dir}")

        if not self._config["classes"]:
            issues.append("No classes found in dataset")
        elif len(self._config["classes"]) < 1:
            issues.append("At least 1 object class required")

        if self._config["total_images"] < 10:
            issues.append(f"Very few images: {self._config['total_images']}")

        return issues

    def save_config(self, output_path: str):
        """Save configuration to file"""
        with open(output_path, 'w') as f:
            json.dump(self._config, f, indent=2)
        logger.info(f"Configuration saved to {output_path}")

    def print_summary(self):
        """Print dataset summary"""
        config = self._config
        print("\n" + "="*60)
        print("📊 FACE DATASET CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Dataset Root: {config['dataset_root']}")
        print(f"Image Directory: {config['image_dir']}")
        print(f"Annotation Directory: {config['annotation_dir']}")
        print(f"Total Images: {config['total_images']:,}")
        print(f"Total Annotations: {config['total_annotations']:,}")
        print(f"Model num_classes: {config['num_classes']}")
        print(f"YOLO Class Range: {config['class_range']}")

        print(f"\n🏷️  Class Distribution:")
        for class_id, count in sorted(config["classes"].items()):
            print(f"  {class_id}: {count:,} annotations")

        print("="*60)
#!/usr/bin/env python3
"""
Dataset Configuration Management
Automatically detects and configures dataset parameters
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse

logger = logging.getLogger(__name__)


class DatasetConfig:
    """Automatically configure dataset parameters"""

    def __init__(self, dataset_root: str, annotation_format: str = "yolo"):
        self.dataset_root = Path(dataset_root)
        self.annotation_format = annotation_format
        self._config = None
        self._analyze_dataset()

    def _analyze_dataset(self):
        """Analyze dataset structure and classes"""
        config = {
            "dataset_root": str(self.dataset_root),
            "annotation_format": self.annotation_format,
            "splits": {},
            "classes": {},
            "total_annotations": 0,
            "total_images": 0
        }

        # Check for standard splits
        for split in ["train", "val", "test"]:
            split_path = self.dataset_root / split
            if split_path.exists():
                images_path = split_path / "images"
                labels_path = split_path / "labels"

                if images_path.exists() and labels_path.exists():
                    split_info = self._analyze_split(images_path, labels_path)
                    config["splits"][split] = split_info
                    config["total_annotations"] += split_info["total_annotations"]
                    config["total_images"] += split_info["total_images"]

                    # Merge class information
                    for class_id, count in split_info["class_counts"].items():
                        if class_id in config["classes"]:
                            config["classes"][class_id] += count
                        else:
                            config["classes"][class_id] = count

        # Calculate num_classes for model
        if config["classes"]:
            max_yolo_class = max(config["classes"].keys())
            # YOLO classes are 0-indexed, model needs background + max_class + 1
            config["num_classes"] = max_yolo_class + 2
            config["class_range"] = [
                min(config["classes"].keys()), max_yolo_class]
        else:
            config["num_classes"] = 2  # Default: background + 1 class
            config["class_range"] = [0, 0]

        self._config = config
        logger.info(
            f"Dataset analysis complete: {config['total_images']} images, {config['num_classes']} classes")

    def _analyze_split(self, images_path: Path, labels_path: Path) -> Dict[str, Any]:
        """Analyze a single dataset split"""
        class_counts = {}
        total_annotations = 0
        image_count = 0

        # Count images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for ext in image_extensions:
            image_count += len(list(images_path.glob(f"*{ext}")))
            image_count += len(list(images_path.glob(f"*{ext.upper()}")))

        # Analyze labels
        if self.annotation_format == "yolo":
            for label_file in labels_path.glob("*.txt"):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0]))
                                class_counts[class_id] = class_counts.get(
                                    class_id, 0) + 1
                                total_annotations += 1
                except Exception as e:
                    logger.warning(f"Error reading {label_file}: {e}")

        return {
            "images_path": str(images_path),
            "labels_path": str(labels_path),
            "total_images": image_count,
            "total_annotations": total_annotations,
            "class_counts": class_counts
        }

    @property
    def num_classes(self) -> int:
        """Get number of classes for model (including background)"""
        return self._config["num_classes"]

    @property
    def class_names(self) -> List[str]:
        """Get class names (if available)"""
        # Default names
        num_classes = self.num_classes
        return ["background"] + [f"class_{i}" for i in range(1, num_classes)]

    def get_split_paths(self, split: str) -> Dict[str, str]:
        """Get paths for a specific split"""
        if split not in self._config["splits"]:
            raise ValueError(
                f"Split '{split}' not found. Available: {list(self._config['splits'].keys())}")

        split_info = self._config["splits"][split]
        return {
            "images": split_info["images_path"],
            "labels": split_info["labels_path"]
        }

    def validate_for_training(self) -> List[str]:
        """Validate dataset configuration for training"""
        issues = []

        # Check required splits
        if "train" not in self._config["splits"]:
            issues.append("Missing 'train' split")

        # Check class distribution
        if not self._config["classes"]:
            issues.append("No classes found in dataset")
        elif len(self._config["classes"]) < 1:
            issues.append("At least 1 object class required")

        # Check for reasonable data amounts
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
        print("📊 DATASET CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Dataset Root: {config['dataset_root']}")
        print(f"Format: {config['annotation_format']}")
        print(f"Total Images: {config['total_images']:,}")
        print(f"Total Annotations: {config['total_annotations']:,}")
        print(f"Model num_classes: {config['num_classes']}")
        print(f"YOLO Class Range: {config['class_range']}")

        print(f"\n📁 Splits:")
        for split, info in config["splits"].items():
            print(
                f"  {split}: {info['total_images']:,} images, {info['total_annotations']:,} annotations")

        print(f"\n🏷️  Class Distribution:")
        for class_id, count in sorted(config["classes"].items()):
            print(f"  YOLO Class {class_id}: {count:,} annotations")

        print("="*60)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for dataset configuration"""
    parser = argparse.ArgumentParser(
        description="Dataset Configuration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m config.dataset_config --dataset processed_data/cattle
  python -m config.dataset_config --dataset processed_data/cattle --save-config dataset_config.json
  python -m config.dataset_config --dataset processed_data/cattle --format yolo
        """
    )

    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to dataset root directory"
    )

    parser.add_argument(
        "--format", "-f",
        default="yolo",
        choices=["yolo", "coco", "voc"],
        help="Annotation format (default: yolo)"
    )

    parser.add_argument(
        "--save-config", "-s",
        help="Save configuration to JSON file"
    )

    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate dataset for training"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output (except errors)"
    )

    return parser


def main():
    """Main CLI function"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    if not args.quiet:
        logging.basicConfig(level=logging.INFO)

    try:
        # Create dataset config
        config = DatasetConfig(args.dataset, args.format)

        if not args.quiet:
            config.print_summary()

        # Validate if requested
        if args.validate:
            issues = config.validate_for_training()
            if issues:
                print(f"\n❌ Validation Issues:")
                for issue in issues:
                    print(f"  - {issue}")
                return 1
            else:
                print(f"\n✅ Dataset validation passed!")

        # Save config if requested
        if args.save_config:
            config.save_config(args.save_config)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

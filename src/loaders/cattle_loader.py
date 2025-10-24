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
from ..utils.box_validation import validate_boxes, clip_boxes
from ..utils.data_analysis import (
    analyze_class_distribution,
    analyze_box_sizes,
    validate_annotations,
    visualize_samples
)

logger = logging.getLogger(__name__)


class CattleDetectionDataset(BaseDetectionLoader):
    """Cattle Detection Data Loader.

    Supports YOLO format:
    - YOLO: images/ and labels/ directories with .txt files
    
    Note: COCO and Pascal VOC formats are not currently supported.
    Please convert your data to YOLO format first.
    """
    
    def _convert_yolo_to_xyxy(self, box, img_w, img_h):
        """Convert YOLO format (xcen, ycen, w, h) to XYXY format.
        Also scales boxes to target image size for better regression learning.
        
        Args:
            box (tensor): Box in YOLO format (xcen, ycen, w, h) normalized to [0, 1]
            img_w (int): Original image width
            img_h (int): Original image height
            
        Returns:
            tensor: Box in XYXY format scaled to target size
        """
        # YOLO box: [xcen, ycen, w, h] normalized
        xcen, ycen, w, h = box
        
        # Convert to pixel coordinates in original image
        xcen = xcen * img_w
        ycen = ycen * img_h
        w = w * img_w
        h = h * img_h
        
        # Convert to XYXY format
        x1 = xcen - w/2
        y1 = ycen - h/2
        x2 = xcen + w/2
        y2 = ycen + h/2
        
        # Scale to target size
        scale_x = self.image_size / img_w
        scale_y = self.image_size / img_h
        
        x1 = x1 * scale_x
        x2 = x2 * scale_x
        y1 = y1 * scale_y
        y2 = y2 * scale_y
        
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

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
        skip_analysis: bool = False,
        max_samples: Optional[int] = None,
        image_dir: Optional[str] = None,
        annotation_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize cattle detection dataset.

        Args:
            root_dir (str): Root directory containing dataset
            split (str): Dataset split ('train', 'val', or 'test')
            image_size (int): Target image size for resizing
            format (str): Annotation format ('yolo', 'coco', or 'pascal')
            transform (Any, optional): Optional transform to apply
            augment (bool): Whether to apply data augmentation
            class_names (List[str], optional): List of class names
            num_classes (int, optional): Number of classes
            image_dir (str, optional): For parallel directory structure, directory containing images
            annotation_dir (str, optional): For parallel directory structure, directory containing annotations
            **kwargs: Additional arguments passed to parent class
        """
        # Store transform for this class
        self.transform = transform

        # Call parent class initialization
        super().__init__(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            augment=augment,
            **kwargs
        )

        # Store input parameters
        self.split = split
        self.format = format.lower()
        self.root_path = Path(root_dir)
        self._input_num_classes = num_classes
        self._input_class_names = class_names

        # Set up directory structure
        self.root_dir = Path(root_dir)
        
        # First check if explicit image_dir and annotation_dir are provided
        if image_dir and annotation_dir:
            # Parallel directory structure
            self.img_dir = self.root_dir / image_dir
            self.label_dir = self.root_dir / annotation_dir
            logger.info(f"Using parallel directory structure with images: {self.img_dir}, annotations: {self.label_dir}")
        else:
            # Try standard YOLO structure with split directories
            self.split_dir = self.root_dir / split
            if (self.split_dir / 'images').exists() and (self.split_dir / 'labels').exists():
                self.img_dir = self.split_dir / 'images'
                self.label_dir = self.split_dir / 'labels'
                logger.info(f"Using standard YOLO directory structure for split: {split}")
            else:
                # Fall back to using the root dir as image dir and a labels subdir
                self.img_dir = self.root_dir 
                self.label_dir = self.root_dir / 'labels'
                logger.info(f"Using root directory for images with labels subdirectory")
        
        # Verify directories exist
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.img_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.label_dir}")
            
        # Support multiple image formats
        img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.img_paths = []
        for ext in img_extensions:
            self.img_paths.extend(list(self.img_dir.glob(ext)))
            self.img_paths.extend(list(self.img_dir.glob(ext.upper())))
        self.img_paths = sorted(self.img_paths)

        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
            
        logger.info(f"Found {len(self.img_paths)} images in {self.img_dir}")

        # Initialize other attributes
        self._image_cache = {}
        self._stats = {'load_errors': 0, 'empty_labels': 0, 'skipped_boxes': 0}
        self.image_size = image_size  # Store target size for box normalization
        
        # Resolve class metadata
        loaded_class_names = self._load_class_names()
        
        # Handle class names
        if self._input_class_names is not None:
            self.class_names = list(self._input_class_names)
            if len(self.class_names) < len(loaded_class_names):
                logger.warning(
                    f"Provided class_names ({len(self.class_names)}) "
                    f"is smaller than detected classes ({len(loaded_class_names)}). "
                    "Using detected classes instead."
                )
                self.class_names = loaded_class_names
        else:
            self.class_names = loaded_class_names

        # Handle number of classes
        if self._input_num_classes is not None:
            self.num_classes = max(int(self._input_num_classes), len(self.class_names))
        else:
            self.num_classes = len(self.class_names)

        # Extend class names if needed
        if len(self.class_names) < self.num_classes:
            start_idx = len(self.class_names)
            self.class_names.extend(
                [f'class_{i}' for i in range(start_idx, self.num_classes)]
            )

        # Validate format
        if self.format not in ['yolo', 'coco', 'pascal']:
            raise ValueError(
                f"Format '{self.format}' not supported. "
                f"Use 'yolo', 'coco', or 'pascal'"
            )

        if self.format != 'yolo':
            raise NotImplementedError(
                f"{self.format.upper()} format is not yet implemented. "
                f"Please convert your dataset to YOLO format. "
                f"Expected structure: {{images/, labels/}} with .txt annotations"
            )

        # Load dataset samples with progress logging
        logger.info("Loading annotations...")
        try:
            self.samples = self._load_annotations()
            logger.info("Successfully loaded annotations.")
        except Exception as e:
            logger.error(f"Error loading annotations: {str(e)}")
            raise
        
        # Apply max_samples limit if specified
        if max_samples is not None:
            logger.info(f"Limiting dataset to {max_samples} samples for smoke test...")
            self.samples = self.samples[:max_samples]
            logger.info(f"Limited dataset to {max_samples} samples")
        
        logger.info(
            f"Final dataset: {len(self.samples)} samples from {self.split} split "
            f"({self.num_classes} classes)"
        )
        
        # Analyze dataset statistics for training split
        if self.split == 'train' and not skip_analysis:
            self.analyze_dataset()
        
        # Store transform before calling parent
        self.transform = transform
        
        # Call parent class __init__
        super().__init__(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            transform=self.transform,
            augment=augment,
            **kwargs
        )
        
        # Store init params
        self.format = format.lower()
        self.split = split
        self.root_path = Path(root_dir)

        # Resolve class metadata
        loaded_class_names = self._load_class_names()
        
        # If explicit class_names provided, ensure it has enough classes
        if class_names is not None:
            self.class_names = list(class_names)
            if len(self.class_names) < len(loaded_class_names):
                logger.warning(
                    f"Provided class_names ({len(self.class_names)}) "
                    f"is smaller than detected classes ({len(loaded_class_names)}). "
                    "Using detected classes instead."
                )
                self.class_names = loaded_class_names
        else:
            self.class_names = loaded_class_names

        # If explicit num_classes provided, ensure it's large enough 
        provided_num_classes = num_classes  # Store input param
        if provided_num_classes is not None:
            self.num_classes = max(int(provided_num_classes), len(self.class_names))
        else:
            self.num_classes = len(self.class_names)

        # If we need more class names than provided, extend with generated names
        if len(self.class_names) < self.num_classes:
            start_idx = len(self.class_names)
            self.class_names.extend(
                [f'class_{i}' for i in range(start_idx, self.num_classes)]
            )

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

        # Verify directories exist
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.img_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.label_dir}")

        self._image_cache = {}  # Cache for loaded images
        self._stats = {'load_errors': 0, 'empty_labels': 0, 'skipped_boxes': 0}
        
        # Load dataset samples
        self.samples = self._load_annotations()

        logger.info(
            f"Loaded {len(self.samples)} samples from {split} split "
            f"({self.num_classes} classes)"
        )
        
        # Analyze dataset statistics for training split if not skipped
        if split == 'train' and not skip_analysis:
            self.analyze_dataset()
        
        # Support multiple image formats
        img_extensions = ('*.jpg', '*.jpeg', '*.png')
        self.img_paths = []
        for ext in img_extensions:
            self.img_paths.extend(self.img_dir.glob(ext))
        self.img_paths = sorted(self.img_paths)

        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        logger.info(f"Initialized {split} dataset with {len(self.img_paths)} images")

    def analyze_dataset(self):
        """
        Perform comprehensive dataset analysis including class distribution,
        box sizes, and annotation quality.
        """
        logger.info("Analyzing dataset...")
        
        # Get all label paths
        label_paths = []
        for img_path in self.img_paths:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                label_paths.append(label_path)
        
        if not label_paths:
            logger.warning(f"No label files found in {self.label_dir}")
            return
            
        # Analyze class distribution
        class_counts = analyze_class_distribution(label_paths)
        self.class_distribution = class_counts
        
        # Analyze box sizes
        box_stats = analyze_box_sizes(label_paths)
        self.box_stats = box_stats
        
        # Validate annotations
        validation_results = validate_annotations(
            label_paths=label_paths,
            img_dir=self.img_dir,
            min_box_size=0.00005  # Reduced threshold for small objects
        )
        self.validation_results = validation_results
        
        # Generate visualization samples
        visualize_samples(
            dataset_path=self.root_dir,
            num_samples=5,
            output_dir=self.root_dir / 'analysis' / 'visualizations'
        )
        
        # Log comprehensive analysis
        logger.info("\nDataset Analysis Summary:")
        logger.info("-------------------------")
        
        # Class distribution summary
        total_boxes = sum(class_counts.values())
        logger.info("\nClass Distribution:")
        for class_id, count in class_counts.items():
            percentage = (count / total_boxes) * 100 if total_boxes > 0 else 0
            logger.info(f"Class {class_id}: {count} boxes ({percentage:.2f}%)")
        
        # Box size summary
        logger.info("\nBox Size Statistics:")
        for stat, value in box_stats.items():
            logger.info(f"{stat}: {value:.6f}")
        
        # Validation issues summary
        logger.info("\nValidation Issues:")
        for issue, count in validation_results.items():
            logger.info(f"{issue}: {count}")
            
        # Provide recommendations
        self._provide_dataset_recommendations()
        
    def _provide_dataset_recommendations(self):
        """
        Provide recommendations based on dataset analysis.
        """
        logger.info("\nRecommendations:")
        logger.info("----------------")
        
        # Class balance recommendations
        total_boxes = sum(self.class_distribution.values())
        if total_boxes > 0:
            class_percentages = {
                cls: (count / total_boxes) * 100 
                for cls, count in self.class_distribution.items()
            }
            
            imbalanced = any(abs(p1 - p2) > 20 
                           for p1 in class_percentages.values() 
                           for p2 in class_percentages.values())
                           
            if imbalanced:
                logger.info("- Class Imbalance Detected:")
                logger.info("  * Consider using weighted loss function")
                logger.info("  * Apply oversampling to minority classes")
                logger.info("  * Use data augmentation for underrepresented classes")
        
        # Small object handling
        if self.box_stats.get('mean_area', 0) < 0.01:
            logger.info("- Small Object Detection:")
            logger.info("  * Consider increasing input resolution")
            logger.info("  * Add feature pyramid levels for small objects")
            logger.info("  * Use anchor sizes matching your box distribution")
        
        # Data quality recommendations
        if self.validation_results.get('invalid_coords', 0) > 0:
            logger.info("- Data Quality Issues:")
            logger.info("  * Clean up invalid coordinates")
            logger.info("  * Verify annotation format consistency")
            
        if self.validation_results.get('too_small', 0) > 0:
            logger.info("- Small Box Filtering:")
            logger.info("  * Review minimum box size threshold")
            logger.info("  * Consider merging very small adjacent boxes")
            
        # General recommendations
        logger.info("- General Improvements:")
        logger.info("  * Verify annotation consistency")
        logger.info("  * Consider additional augmentation techniques")
        logger.info("  * Review box size distribution for anchor optimization")



    def _load_class_names(self) -> List[str]:
        """Load class names from data.yaml or classes.txt, or auto-detect from labels."""
        # Try data.yaml first
        yaml_path = self.root_path / 'data.yaml'
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    names = data['names']
                    # Handle both dict and list formats
                    if isinstance(names, dict):
                        names = [names[i] for i in sorted(names.keys())]
                    if isinstance(names, list) and len(names) > 0:
                        return names

        # Try classes.txt
        classes_path = self.root_path / 'classes.txt'
        if classes_path.exists():
            with open(classes_path, 'r') as f:
                names = [line.strip() for line in f if line.strip()]
                if len(names) > 0:
                    return names

        # Auto-detect number of classes from labels
        max_class_id = -1
        label_patterns = [
            f"{self.root_path}/{self.split}/labels/**/*.txt",
            f"{self.root_path}/labels/{self.split}/**/*.txt",
            f"{self.root_path}/{self.label_dir}/**/*.txt"  # Add this pattern for custom label dirs
        ]
        
        for pattern in label_patterns:
            try:
                # Use recursive glob to find all txt files in subdirs
                import glob
                for label_file in glob.glob(pattern, recursive=True):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:  # Read all lines to find highest class ID
                                try:
                                    class_id = int(line.split()[0])
                                    max_class_id = max(max_class_id, class_id)
                                except (ValueError, IndexError):
                                    continue
                    except Exception:
                        continue
            except Exception:
                continue

        # If max_class_id is found, use it to set num_classes
        if max_class_id >= 0:
            num_classes = max_class_id + 1
        else:
            # If no labels found, check if num_classes was provided in config
            num_classes = self._input_num_classes if self._input_num_classes else 2

        class_names = [str(i) for i in range(num_classes)]

        if self.split == 'train':  # Only log warning once for training set
            logger.warning(
                f"No class names found in data.yaml or classes.txt. "
                f"Auto-detected {num_classes} classes from labels: {class_names}"
            )
        return class_names

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
        stats = {'total_images': 0, 'skipped_images': 0, 'total_boxes': 0, 'filtered_boxes': 0}

        # Use the configured image and label directories
        logger.info(f"Using configured directories - Images: {self.img_dir}, Labels: {self.label_dir}")

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.img_dir.glob(f'*{ext}'))
            image_files.extend(self.img_dir.glob(f'*{ext.upper()}'))

        logger.info(f"Found {len(image_files)} images in {self.img_dir}")

        for img_path in sorted(image_files):
            # Get corresponding label file
            label_path = self.label_dir / f"{img_path.stem}.txt"

            # Load annotations from label file
            boxes = []
            labels = []

            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            parts = line.split()
                            if len(parts) < 5:
                                logger.warning(f"Skipping invalid annotation in {label_path}:{line_num} - not enough values")
                                continue

                            # YOLO format: class_id x_center y_center width height
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            # Validate values
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                                logger.warning(f"Skipping invalid box in {label_path}:{line_num} - values out of range [0,1]")
                                continue

                            if class_id >= self.num_classes:
                                logger.warning(f"Skipping invalid class ID in {label_path}:{line_num} - class {class_id} >= num_classes {self.num_classes}")
                                continue

                            # Convert to [x_min, y_min, x_max, y_max] format
                            x_min = max(0.0, x_center - width / 2)
                            y_min = max(0.0, y_center - height / 2)
                            x_max = min(1.0, x_center + width / 2)
                            y_max = min(1.0, y_center + height / 2)

                            # Validate box after conversion
                            if x_max <= x_min or y_max <= y_min:
                                logger.warning(f"Skipping invalid box in {label_path}:{line_num} - zero or negative area")
                                continue

                            # Calculate box area
                            box_area = (x_max - x_min) * (y_max - y_min)
                            if box_area < 0.0001:  # Skip extremely small boxes
                                logger.warning(f"Skipping very small box in {label_path}:{line_num} - area = {box_area:.6f}")
                                continue

                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)

                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing annotation in {label_path}:{line_num} - {str(e)}")
                            continue

            # Get image size if we need it later
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                logger.warning(f"Could not get size for {img_path}: {e}")
                width, height = 640, 640  # Default fallback size

            # Create sample dict with proper tensor types
            sample = {
                'image_path': str(img_path),
                'boxes': boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
                'labels': labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([len(samples)], dtype=torch.int64),
                'original_size': torch.tensor([height, width], dtype=torch.int64)
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

        # Convert YOLO normalized boxes to XYXY format
        if isinstance(boxes, list) and len(boxes) > 0:
            boxes_converted = []
            for box in boxes:
                # Convert single box using helper method
                box_xyxy = self._convert_yolo_to_xyxy(
                    torch.tensor(box), 
                    img_width, 
                    img_height
                )
                boxes_converted.append(box_xyxy)
            boxes = torch.stack(boxes_converted)
            labels = torch.tensor(labels, dtype=torch.int64)
        elif isinstance(boxes, torch.Tensor):
            if boxes.numel() > 0:  # If tensor is not empty
                boxes = boxes.clone()  # Clone to avoid modifying original
                boxes[:, [0, 2]] *= img_width
                boxes[:, [1, 3]] *= img_height
            labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

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

        # Apply transform
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

import torch
import logging
from typing import Dict, List, Any, Union, Tuple, Optional
import os


def analyze_dataset_classes(dataset_path: str, annotation_format: str = "yolo") -> Dict[str, Any]:
    """
    Analyze dataset to determine class configuration

    Args:
        dataset_path: Path to directory containing label files
        annotation_format: Format of annotations ('yolo', 'coco', 'voc')

    Returns:
        Dict with class analysis information
    """
    class_counts = {}
    total_annotations = 0
    file_count = 0

    if annotation_format == "yolo":
        extension = ".txt"
    elif annotation_format == "coco":
        extension = ".json"
    elif annotation_format == "voc":
        extension = ".xml"
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")

    if not os.path.exists(dataset_path):
        logging.warning(f"Dataset path does not exist: {dataset_path}")
        return {
            "num_classes": 2,  # default: background + 1 class
            "class_counts": {0: 0},
            "max_class": 0,
            "min_class": 0,
            "total_annotations": 0,
            "total_files": 0,
            "error": f"Path not found: {dataset_path}"
        }

    try:
        for filename in os.listdir(dataset_path):
            if not filename.endswith(extension):
                continue

            file_count += 1
            filepath = os.path.join(dataset_path, filename)

            if annotation_format == "yolo":
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # class_id + 4 bbox coords
                                class_id = int(float(parts[0]))
                                class_counts[class_id] = class_counts.get(
                                    class_id, 0) + 1
                                total_annotations += 1
                except Exception as e:
                    logging.warning(f"Error reading {filepath}: {e}")
                    continue
            # Add support for other formats if needed

    except Exception as e:
        logging.error(f"Error analyzing dataset: {e}")
        return {
            "num_classes": 2,
            "class_counts": {0: 0},
            "max_class": 0,
            "min_class": 0,
            "total_annotations": 0,
            "total_files": 0,
            "error": str(e)
        }

    if not class_counts:
        return {
            "num_classes": 2,
            "class_counts": {},
            "max_class": 0,
            "min_class": 0,
            "total_annotations": 0,
            "total_files": file_count,
            "warning": "No annotations found"
        }

    min_class = min(class_counts.keys())
    max_class = max(class_counts.keys())

    # For Faster R-CNN: need background class (0) + object classes
    # If YOLO classes start from 0, model needs num_classes = max_class + 1 + 1 (for background)
    # But since we shift YOLO classes by +1 in dataset.py, model needs max_class + 2
    num_classes = max_class + 2  # +1 for shift, +1 for background

    return {
        "num_classes": num_classes,
        "class_counts": class_counts,
        "max_class": max_class,
        "min_class": min_class,
        "total_annotations": total_annotations,
        "total_files": file_count,
        "classes_found": sorted(class_counts.keys())
    }


def validate_boxes(boxes: torch.Tensor) -> Tuple[bool, str]:
    """
    Comprehensive bounding box validation

    Args:
        boxes: torch.Tensor of shape [N, 4] with format [x1, y1, x2, y2]

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(boxes, torch.Tensor):
        return False, "Boxes must be a torch.Tensor"

    if boxes.numel() == 0:  # Empty boxes are valid
        return True, ""

    if boxes.dim() != 2 or boxes.size(1) != 4:
        return False, f"Boxes must have shape [N, 4], got {boxes.shape}"

    # Check for NaN or infinite values
    if not torch.isfinite(boxes).all():
        return False, "Boxes contain NaN or infinite values"

    # Check coordinate validity: x1 < x2 and y1 < y2
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if not torch.all(x2 > x1):
        invalid_idx = torch.where(x2 <= x1)[0]
        return False, f"Invalid x-coordinates (x2 <= x1) at indices: {invalid_idx.tolist()}"

    if not torch.all(y2 > y1):
        invalid_idx = torch.where(y2 <= y1)[0]
        return False, f"Invalid y-coordinates (y2 <= y1) at indices: {invalid_idx.tolist()}"

    # Check for negative coordinates
    if torch.any(boxes < 0):
        return False, "Boxes contain negative coordinates"

    # Check for extremely small boxes (likely annotation errors)
    box_widths = x2 - x1
    box_heights = y2 - y1
    min_size = 1.0  # minimum 1 pixel

    if torch.any(box_widths < min_size) or torch.any(box_heights < min_size):
        small_boxes = torch.where(
            (box_widths < min_size) | (box_heights < min_size))[0]
        return False, f"Boxes too small (< {min_size} pixels) at indices: {small_boxes.tolist()}"

    return True, ""


def validate_labels(labels: torch.Tensor, num_classes: int = None) -> Tuple[bool, str]:
    """
    Validate class labels for object detection

    Args:
        labels: torch.Tensor of shape [N] with class indices
        num_classes: int, number of classes in model (including background). 
                    If None, will validate basic format only

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(labels, torch.Tensor):
        return False, "Labels must be a torch.Tensor"

    if labels.numel() == 0:  # Empty labels are valid
        return True, ""

    if labels.dim() != 1:
        return False, f"Labels must be 1D tensor, got shape {labels.shape}"

    if labels.dtype != torch.int64:
        return False, f"Labels must be int64, got {labels.dtype}"

    # Check for negative class indices (always invalid)
    min_class = labels.min().item()
    if min_class < 0:
        invalid_idx = torch.where(labels < 0)[0]
        return False, f"Invalid negative class labels at indices: {invalid_idx.tolist()}. Found: {labels[invalid_idx].tolist()}"

    # If num_classes is provided, validate against it
    if num_classes is not None:
        # For Faster R-CNN: background=0, objects=1,2,3...
        # So valid range is [1, num_classes-1] for objects
        max_class = labels.max().item()

        if min_class < 1:
            invalid_idx = torch.where(labels < 1)[0]
            return False, f"Invalid class labels < 1 at indices: {invalid_idx.tolist()}. Found: {labels[invalid_idx].tolist()}"

        if max_class >= num_classes:
            invalid_idx = torch.where(labels >= num_classes)[0]
            return False, f"Invalid class labels >= {num_classes} at indices: {invalid_idx.tolist()}. Found: {labels[invalid_idx].tolist()}"

    return True, ""


def infer_num_classes_from_targets(targets: List[Dict[str, Any]]) -> int:
    """
    Infer the number of classes from target labels

    Args:
        targets: List of target dictionaries with 'labels' key

    Returns:
        int: Number of classes (including background class 0)
    """
    all_labels = []

    for target in targets:
        if 'labels' in target and target['labels'].numel() > 0:
            labels = target['labels']
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(labels)

    if not all_labels:
        # Default: background + 1 object class
        return 2

    max_class = max(all_labels)
    # num_classes = max_class + 1 (to include background class 0)
    return max_class + 1


def validate_targets(targets: List[Dict[str, Any]], num_classes: int = None) -> Tuple[bool, str]:
    """
    Comprehensive validation of training targets

    Args:
        targets: List of target dictionaries
        num_classes: Number of classes in model (including background).
                    If None, will infer from targets

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        if not targets:
            return True, ""

        # Infer num_classes if not provided
        if num_classes is None:
            num_classes = infer_num_classes_from_targets(targets)

        for i, target in enumerate(targets):
            if not isinstance(target, dict):
                return False, f"Target {i} is not a dictionary"

            if 'boxes' not in target or 'labels' not in target:
                return False, f"Target {i} missing 'boxes' or 'labels' keys"

            boxes = target['boxes']
            labels = target['labels']

            # Validate boxes
            boxes_valid, boxes_error = validate_boxes(boxes)
            if not boxes_valid:
                return False, f"Target {i} boxes invalid: {boxes_error}"

            # Validate labels
            labels_valid, labels_error = validate_labels(labels, num_classes)
            if not labels_valid:
                return False, f"Target {i} labels invalid: {labels_error}"

            # Check consistency between boxes and labels
            if boxes.size(0) != labels.size(0):
                return False, f"Target {i}: boxes ({boxes.size(0)}) and labels ({labels.size(0)}) count mismatch"

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"

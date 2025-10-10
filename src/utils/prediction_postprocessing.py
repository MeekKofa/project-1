"""
Robust Prediction Post-Processing Module.

This module provides comprehensive utilities for cleaning up model predictions:
- Non-Maximum Suppression (NMS) to remove duplicate boxes
- Confidence thresholding to filter low-quality predictions
- Box validation to remove invalid/out-of-bound boxes
- IoU-based filtering to remove overlapping boxes
- Aspect ratio filtering to remove unrealistic boxes

Purpose: Fix the issue of too many bounding boxes being shown in predictions.
"""

import torch
from typing import Dict, List, Tuple, Optional
from torchvision.ops import nms, box_iou


def validate_boxes(
    boxes: torch.Tensor,
    image_width: int,
    image_height: int,
    min_box_size: int = 10,
    max_aspect_ratio: float = 10.0
) -> torch.Tensor:
    """
    Validate bounding boxes and return mask of valid boxes.

    Args:
        boxes: Tensor of shape [N, 4] in xyxy format
        image_width: Image width in pixels
        image_height: Image height in pixels
        min_box_size: Minimum box dimension (width or height) in pixels
        max_aspect_ratio: Maximum allowed width/height or height/width ratio

    Returns:
        Boolean mask of shape [N] indicating valid boxes
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes.device)

    # Clamp boxes to image boundaries
    boxes_clamped = boxes.clone()
    boxes_clamped[:, [0, 2]] = boxes_clamped[:, [0, 2]].clamp(0, image_width)
    boxes_clamped[:, [1, 3]] = boxes_clamped[:, [1, 3]].clamp(0, image_height)

    # Calculate box dimensions
    widths = boxes_clamped[:, 2] - boxes_clamped[:, 0]
    heights = boxes_clamped[:, 3] - boxes_clamped[:, 1]

    # Filter criteria
    valid_size = (widths >= min_box_size) & (heights >= min_box_size)
    valid_area = (widths * heights) > 0

    # Check aspect ratios (width/height and height/width)
    aspect_ratios = widths / (heights + 1e-6)
    valid_aspect = (aspect_ratios <= max_aspect_ratio) & (
        aspect_ratios >= 1.0 / max_aspect_ratio)

    # Check if boxes are within image bounds
    valid_bounds = (
        (boxes_clamped[:, 0] < image_width) &
        (boxes_clamped[:, 1] < image_height) &
        (boxes_clamped[:, 2] > 0) &
        (boxes_clamped[:, 3] > 0)
    )

    # Combine all criteria
    valid_mask = valid_size & valid_area & valid_aspect & valid_bounds

    return valid_mask


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_detections: int = 300
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Non-Maximum Suppression per class to remove duplicate boxes.

    Args:
        boxes: Tensor of shape [N, 4] in xyxy format
        scores: Tensor of shape [N] with confidence scores
        labels: Tensor of shape [N] with class labels
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score to keep a detection
        max_detections: Maximum number of detections to keep

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels)
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    # First filter by score threshold
    score_mask = scores >= score_threshold
    if not score_mask.any():
        device = boxes.device
        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), device=device),
            torch.empty((0,), dtype=torch.int64, device=device)
        )

    boxes = boxes[score_mask]
    scores = scores[score_mask]
    labels = labels[score_mask]

    # Apply NMS per class
    unique_labels = labels.unique()

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    for label in unique_labels:
        label_mask = labels == label
        label_boxes = boxes[label_mask]
        label_scores = scores[label_mask]

        # Apply NMS for this class
        keep_indices = nms(label_boxes, label_scores, iou_threshold)

        keep_boxes.append(label_boxes[keep_indices])
        keep_scores.append(label_scores[keep_indices])
        keep_labels.append(torch.full((keep_indices.numel(),),
                           label, dtype=torch.int64, device=labels.device))

    if not keep_boxes:
        device = boxes.device
        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), device=device),
            torch.empty((0,), dtype=torch.int64, device=device)
        )

    # Concatenate all classes
    boxes = torch.cat(keep_boxes, dim=0)
    scores = torch.cat(keep_scores, dim=0)
    labels = torch.cat(keep_labels, dim=0)

    # Sort by score and keep top-k
    if boxes.size(0) > max_detections:
        _, top_indices = torch.topk(scores, k=max_detections)
        boxes = boxes[top_indices]
        scores = scores[top_indices]
        labels = labels[top_indices]

    return boxes, scores, labels


def remove_overlapping_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.7
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove boxes that significantly overlap with higher-scoring boxes.

    Args:
        boxes: Tensor of shape [N, 4] in xyxy format
        scores: Tensor of shape [N] with confidence scores
        labels: Tensor of shape [N] with class labels
        iou_threshold: IoU threshold for considering boxes as overlapping

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels)
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    # Sort by score descending
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]

    keep_mask = torch.ones(boxes.size(
        0), dtype=torch.bool, device=boxes.device)

    # For each box, check if it overlaps significantly with any higher-scoring box
    for i in range(boxes.size(0)):
        if not keep_mask[i]:
            continue

        # Compute IoU with all remaining boxes
        ious = box_iou(boxes[i:i+1], boxes[i+1:]).squeeze(0)

        # Mark boxes with high IoU as duplicates
        overlap_mask = ious > iou_threshold
        keep_mask[i+1:][overlap_mask] = False

    return boxes[keep_mask], scores[keep_mask], labels[keep_mask]


def filter_outside_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_width: int,
    image_height: int,
    margin: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter out boxes that are mostly outside the image boundaries.

    Args:
        boxes: Tensor of shape [N, 4] in xyxy format
        scores: Tensor of shape [N] with confidence scores
        labels: Tensor of shape [N] with class labels
        image_width: Image width in pixels
        image_height: Image height in pixels
        margin: Fraction of box that must be inside image (0.1 = 10%)

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels)
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    # Clamp boxes to image boundaries
    boxes_clamped = boxes.clone()
    boxes_clamped[:, [0, 2]] = boxes_clamped[:, [0, 2]].clamp(0, image_width)
    boxes_clamped[:, [1, 3]] = boxes_clamped[:, [1, 3]].clamp(0, image_height)

    # Calculate original and clamped areas
    original_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    clamped_areas = (boxes_clamped[:, 2] - boxes_clamped[:, 0]) * \
        (boxes_clamped[:, 3] - boxes_clamped[:, 1])

    # Keep boxes where at least margin% of area is inside image
    area_ratios = clamped_areas / (original_areas + 1e-6)
    keep_mask = area_ratios >= margin

    return boxes[keep_mask], scores[keep_mask], labels[keep_mask]


def robust_postprocess_predictions(
    predictions: Dict[str, torch.Tensor],
    image_width: int,
    image_height: int,
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    overlap_iou_threshold: float = 0.7,
    max_detections: int = 100,
    min_box_size: int = 10,
    max_aspect_ratio: float = 10.0,
    outside_margin: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Comprehensive post-processing pipeline for model predictions.

    This function applies multiple filtering stages to clean up predictions:
    1. Confidence thresholding
    2. Box validation (size, aspect ratio, bounds)
    3. Non-Maximum Suppression (NMS)
    4. Overlap filtering
    5. Outside-image filtering

    Args:
        predictions: Dict with 'boxes', 'scores', 'labels' tensors
        image_width: Image width in pixels
        image_height: Image height in pixels
        conf_threshold: Minimum confidence score (0.0-1.0)
        nms_iou_threshold: IoU threshold for NMS (typically 0.4-0.5)
        overlap_iou_threshold: IoU threshold for removing overlaps (typically 0.6-0.8)
        max_detections: Maximum number of detections to keep
        min_box_size: Minimum box dimension in pixels
        max_aspect_ratio: Maximum allowed aspect ratio
        outside_margin: Minimum fraction of box that must be inside image

    Returns:
        Dict with filtered 'boxes', 'scores', 'labels' tensors
    """
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']

    if boxes.numel() == 0:
        return predictions

    # Stage 1: Validate boxes (size, aspect ratio, bounds)
    valid_mask = validate_boxes(
        boxes, image_width, image_height,
        min_box_size=min_box_size,
        max_aspect_ratio=max_aspect_ratio
    )
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    labels = labels[valid_mask]

    if boxes.numel() == 0:
        device = predictions['boxes'].device
        return {
            'boxes': torch.empty((0, 4), device=device),
            'scores': torch.empty((0,), device=device),
            'labels': torch.empty((0,), dtype=torch.int64, device=device)
        }

    # Stage 2: Apply NMS (removes duplicates)
    boxes, scores, labels = apply_nms(
        boxes, scores, labels,
        iou_threshold=nms_iou_threshold,
        score_threshold=conf_threshold,
        max_detections=max_detections
    )

    if boxes.numel() == 0:
        device = predictions['boxes'].device
        return {
            'boxes': torch.empty((0, 4), device=device),
            'scores': torch.empty((0,), device=device),
            'labels': torch.empty((0,), dtype=torch.int64, device=device)
        }

    # Stage 3: Remove highly overlapping boxes
    boxes, scores, labels = remove_overlapping_boxes(
        boxes, scores, labels,
        iou_threshold=overlap_iou_threshold
    )

    # Stage 4: Filter boxes mostly outside image
    boxes, scores, labels = filter_outside_boxes(
        boxes, scores, labels,
        image_width, image_height,
        margin=outside_margin
    )

    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels
    }


__all__ = [
    'validate_boxes',
    'apply_nms',
    'remove_overlapping_boxes',
    'filter_outside_boxes',
    'robust_postprocess_predictions'
]

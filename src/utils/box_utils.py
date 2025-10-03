"""
Box utility functions for object detection.

Provides common box operations like IoU, NMS, format conversion, etc.
"""

import torch
from typing import Tuple


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute the area of boxes.

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format

    Returns:
        [N] areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format
        boxes2: [M, 4] in (x1, y1, x2, y2) format

    Returns:
        [N, M] IoU matrix
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute union
    union = area1[:, None] + area2 - inter

    # Compute IoU
    iou = inter / union
    return iou


def sanitize_boxes(
    boxes: torch.Tensor,
    image_size: Tuple[int, int] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Sanitize boxes to ensure validity.

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format
        image_size: (height, width) to clamp boxes
        eps: Minimum box dimension

    Returns:
        Sanitized boxes [N, 4]
    """
    if boxes.numel() == 0:
        return boxes

    # Handle NaN and inf
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1e4, neginf=-1e4)

    # Ensure x1 <= x2 and y1 <= y2
    x1 = torch.min(boxes[:, 0], boxes[:, 2])
    y1 = torch.min(boxes[:, 1], boxes[:, 3])
    x2 = torch.max(boxes[:, 0], boxes[:, 2])
    y2 = torch.max(boxes[:, 1], boxes[:, 3])

    # Clamp to image bounds if provided
    if image_size is not None:
        h, w = image_size
        x1 = x1.clamp(min=0, max=w)
        y1 = y1.clamp(min=0, max=h)
        x2 = x2.clamp(min=0, max=w)
        y2 = y2.clamp(min=0, max=h)

    # Ensure minimum dimension
    x2 = torch.clamp(x2, min=x1 + eps)
    y2 = torch.clamp(y2, min=y1 + eps)

    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        boxes: [N, 4] in (cx, cy, w, h) format

    Returns:
        [N, 4] in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format

    Returns:
        [N, 4] in (cx, cy, w, h) format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def clip_boxes_to_image(
    boxes: torch.Tensor,
    image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Clip boxes to image boundaries.

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format
        image_size: (height, width)

    Returns:
        Clipped boxes [N, 4]
    """
    h, w = image_size
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h)
    return boxes


def remove_small_boxes(
    boxes: torch.Tensor,
    min_size: float = 1.0
) -> torch.Tensor:
    """
    Remove boxes smaller than min_size.

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format
        min_size: Minimum box dimension

    Returns:
        Boolean mask [N] of boxes to keep
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    return keep

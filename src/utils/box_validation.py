"""
Box validation utilities for object detection.
"""

import numpy as np
import torch
import logging
from typing import Union, Tuple, List, Optional

logger = logging.getLogger(__name__)

def validate_boxes(
    boxes: Union[np.ndarray, torch.Tensor],
    image_size: Optional[Tuple[int, int]] = None,
    min_area: float = 0.0001,
    min_visibility: float = 0.1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Validate and filter detection boxes.

    Args:
        boxes: Array of boxes in [x_min, y_min, x_max, y_max] format
              Values should be normalized to [0,1] if image_size is None
        image_size: Optional (height, width) tuple for denormalized coordinates
        min_area: Minimum relative box area (normalized)
        min_visibility: Minimum box visibility when clipped to image bounds

    Returns:
        Filtered array of valid boxes
    """
    if len(boxes) == 0:
        return boxes

    # Convert torch tensor to numpy if needed
    is_torch = isinstance(boxes, torch.Tensor)
    if is_torch:
        boxes = boxes.detach().cpu().numpy()

    # Handle denormalized coordinates
    if image_size is not None:
        h, w = image_size
        boxes = boxes.copy()
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h

    valid_boxes = []
    
    for box_idx, box in enumerate(boxes):
        try:
            x_min, y_min, x_max, y_max = box[:4]
            
            # Basic coordinate validation 
            if not (0 <= x_min <= 1 and 0 <= y_min <= 1 and 
                   0 <= x_max <= 1 and 0 <= y_max <= 1):
                logger.debug(f"Box {box_idx}: coordinates out of [0,1] range")
                continue
                
            # Check proper ordering
            if x_max <= x_min or y_max <= y_min:
                logger.debug(f"Box {box_idx}: invalid dimensions (x_max <= x_min or y_max <= y_min)")
                continue

            # Calculate box area
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            
            if area < min_area:
                logger.debug(f"Box {box_idx}: area {area:.6f} below minimum {min_area}")
                continue
                
            # Calculate visibility after clipping
            x_min_clip = max(0, x_min)
            y_min_clip = max(0, y_min) 
            x_max_clip = min(1, x_max)
            y_max_clip = min(1, y_max)
            
            clipped_area = (x_max_clip - x_min_clip) * (y_max_clip - y_min_clip)
            visibility = clipped_area / area
            
            if visibility < min_visibility:
                logger.debug(f"Box {box_idx}: visibility {visibility:.3f} below minimum {min_visibility}")
                continue

            # Convert back to absolute coordinates
            if image_size is not None:
                box[[0,2]] *= w
                box[[1,3]] *= h
                
            valid_boxes.append(box)
            
        except Exception as e:
            logger.warning(f"Error validating box {box_idx}: {str(e)}")
            continue
            
    valid_boxes = np.array(valid_boxes) if valid_boxes else np.zeros((0, boxes.shape[1]))
    
    # Convert back to torch if input was torch tensor
    if is_torch:
        valid_boxes = torch.from_numpy(valid_boxes)
        
    return valid_boxes

def clip_boxes(
    boxes: Union[np.ndarray, torch.Tensor],
    image_size: Optional[Tuple[int, int]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Clip boxes to image boundaries.
    
    Args:
        boxes: Array of boxes in [x_min, y_min, x_max, y_max] format
              Values should be normalized to [0,1] if image_size is None
        image_size: Optional (height, width) tuple for denormalized coordinates
        
    Returns:
        Array of clipped boxes
    """
    if len(boxes) == 0:
        return boxes
        
    is_torch = isinstance(boxes, torch.Tensor)
    if is_torch:
        boxes = boxes.detach().cpu().numpy()
        
    boxes = boxes.copy()
    
    if image_size is not None:
        h, w = image_size
        boxes[:, [0, 2]] /= w  
        boxes[:, [1, 3]] /= h

    # Clip to [0, 1]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 1)
    
    if image_size is not None:
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        
    if is_torch:
        boxes = torch.from_numpy(boxes)
        
    return boxes
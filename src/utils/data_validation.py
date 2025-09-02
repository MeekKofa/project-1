import torch
import logging
from typing import Dict, List, Any, Union, Tuple

def validate_boxes(boxes: torch.Tensor) -> bool:
    """Validate bounding boxes format and values"""
    if not isinstance(boxes, torch.Tensor):
        return False
    if boxes.dim() != 2 or boxes.size(1) != 4:
        return False
    if not torch.all((boxes[:, 2:] - boxes[:, :2]) > 0):
        return False
    return True

def validate_targets(targets: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Validate training targets"""
    try:
        for t in targets:
            if not isinstance(t, dict):
                return False, "Target is not a dictionary"
            if 'boxes' not in t or 'labels' not in t:
                return False, "Missing boxes or labels"
            if not validate_boxes(t['boxes']):
                return False, "Invalid box format"
        return True, ""
    except Exception as e:
        return False, str(e)

import torch
import logging

def validate_prediction_shape(pred):
    """Validate prediction tensor shape"""
    if not isinstance(pred, torch.Tensor):
        logging.error(f"Expected torch.Tensor, got {type(pred)}")
        return False
    if pred.dim() != 4:  # [B, C, H, W]
        logging.error(f"Expected 4D tensor, got shape {pred.shape}")
        return False
    return True

def validate_target_boxes(boxes):
    """Validate target boxes shape"""
    if not isinstance(boxes, torch.Tensor):
        return False
    if boxes.dim() != 2 or boxes.size(1) != 4:
        logging.error(f"Invalid boxes shape: {boxes.shape}")
        return False
    return True

import torch
import logging
from typing import List, Dict, Any, Tuple

def validate_and_fix_tensors(images: torch.Tensor, targets: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, Any]], bool]:
    """Validate and fix tensor dimensions and shapes"""
    try:
        # Fix image dimensions
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4:
            logging.error(f"Expected 4D tensor for images, got {images.dim()}D")
            return None, None, False
            
        # Validate and fix target boxes
        for idx, target in enumerate(targets):
            boxes = target.get('boxes')
            if boxes is None:
                logging.error(f"No boxes found in target {idx}")
                return None, None, False
            if boxes.dim() == 1:
                boxes = boxes.view(-1, 4)
                target['boxes'] = boxes
            if boxes.size(-1) != 4:
                logging.error(f"Invalid boxes dimensions in target {idx}: {boxes.shape}")
                return None, None, False
                
        return images, targets, True
        
    except Exception as e:
        logging.error(f"Tensor validation error: {str(e)}")
        return None, None, False

def validate_model_output(predictions: List[torch.Tensor], device=None) -> bool:
    """Validate model output dimensions and device"""
    try:
        for i, pred in enumerate(predictions):
            if pred.dim() != 4:
                logging.error(f"Invalid prediction dimensions at index {i}: {pred.shape}")
                return False
            if device and pred.device != device:
                logging.error(f"Prediction at index {i} is not on the correct device")
                return False
            if not torch.isfinite(pred).all():
                logging.error(f"Non-finite values in prediction at index {i}")
                return False
        return True
    except Exception as e:
        logging.error(f"Prediction validation error: {str(e)}")
        return False

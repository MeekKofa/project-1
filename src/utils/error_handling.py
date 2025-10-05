import logging
import torch
from typing import Dict, Any, List, Union, Optional

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def validate_model_inputs(images: torch.Tensor, targets: List[Dict[str, Any]]) -> bool:
    """Validate model inputs before forward pass"""
    try:
        if not isinstance(images, torch.Tensor):
            logging.error(f"Images must be tensor, got {type(images)}")
            return False
            
        if images.dim() != 4:
            logging.error(f"Images must be 4D (B,C,H,W), got {images.dim()}D")
            return False
            
        for i, target in enumerate(targets):
            if not isinstance(target, dict):
                logging.error(f"Target {i} must be dict, got {type(target)}")
                return False
                
            if 'boxes' not in target or 'labels' not in target:
                logging.error(f"Target {i} missing boxes or labels")
                return False
                
            if len(target['boxes']) != len(target['labels']):
                logging.error(f"Target {i} has mismatched boxes and labels")
                return False
                
        return True
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return False

def validate_loss(loss: torch.Tensor) -> bool:
    """Check if loss value is valid"""
    return bool(torch.isfinite(loss).all().item())

def recover_from_error(e: Exception, device: torch.device) -> torch.Tensor:
    """Handle errors during loss computation"""
    logging.error(f"Error during computation: {str(e)}")
    return torch.tensor(float('inf'), device=device)

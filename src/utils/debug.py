import torch
import logging
from typing import Any, Dict, List, Union

def setup_debug_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )

def inspect_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Print detailed tensor information"""
    logging.debug(f"\n{name}:")
    logging.debug(f"Shape: {tensor.shape}")
    logging.debug(f"Type: {tensor.dtype}")
    logging.debug(f"Device: {tensor.device}")
    if torch.isfinite(tensor).all():
        logging.debug(f"Range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    else:
        logging.debug("Warning: Tensor contains non-finite values")

def inspect_targets(targets: List[Dict[str, Any]]) -> None:
    """Inspect target dictionary contents"""
    for i, target in enumerate(targets):
        logging.debug(f"\nTarget {i}:")
        for k, v in target.items():
            if isinstance(v, torch.Tensor):
                inspect_tensor(v, f"target[{i}][{k}]")
            else:
                logging.debug(f"{k}: {type(v)}")

def inspect_predictions(predictions: List[torch.Tensor], name: str = "predictions") -> None:
    """Inspect model predictions"""
    logging.debug(f"\n{name}:")
    for i, pred in enumerate(predictions):
        inspect_tensor(pred, f"prediction[{i}]")
        if pred.dim() != 4:
            logging.error(f"Invalid prediction shape at index {i}: expected 4D tensor, got {pred.dim()}D")

def validate_shapes(pred: torch.Tensor, target: Dict[str, torch.Tensor]) -> bool:
    """Validate tensor shapes match expectations"""
    try:
        B, C, H, W = pred.shape
        boxes = target.get('boxes')
        if boxes is None:
            logging.error("Missing boxes in target")
            return False
            
        if boxes.dim() != 2 or boxes.size(1) != 4:
            logging.error(f"Invalid boxes shape: {boxes.shape}")
            return False
            
        return True
    except Exception as e:
        logging.error(f"Shape validation error: {str(e)}")
        return False

def debug_loss_computation(loss: torch.Tensor, pred: torch.Tensor, target: Dict[str, torch.Tensor]) -> None:
    """Debug loss computation issues"""
    logging.debug("\nLoss computation debug:")
    logging.debug(f"Loss value: {loss.item() if torch.isfinite(loss) else 'Non-finite'}")
    inspect_tensor(pred, "Predictions")
    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            inspect_tensor(v, f"Target[{k}]")

def debug_batch_dimensions(images, targets, batch_idx):
    """Debug tensor dimensions in batch"""
    logging.debug(f"\nBatch {batch_idx} dimensions:")
    if isinstance(images, torch.Tensor):
        inspect_tensor(images, "images")
    elif isinstance(images, (list, tuple)):
        logging.debug(f"Images is {type(images)} of length {len(images)}")
        for i, img in enumerate(images):
            inspect_tensor(img, f"image[{i}]")
    
    if isinstance(targets, (list, dict)):
        for i, t in enumerate(targets):
            logging.debug(f"\nTarget {i}:")
            if 'boxes' in t:
                inspect_tensor(t['boxes'], f"boxes[{i}]")
            if 'labels' in t:
                inspect_tensor(t['labels'], f"labels[{i}]")

def debug_tensor_shapes(name: str, tensor: torch.Tensor) -> None:
    """Log detailed tensor shape information"""
    try:
        shape_info = f"{name} shape: {tensor.shape}"
        if tensor.dim() > 0:
            shape_info += f", dtype: {tensor.dtype}, device: {tensor.device}"
            shape_info += f", range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]"
        logging.debug(shape_info)
    except Exception as e:
        logging.error(f"Error inspecting tensor {name}: {str(e)}")

def log_model_outputs(outputs, detections, batch_idx):
    """Debug log model outputs"""
    try:
        box_preds, cls_preds = outputs
        logging.debug(f"Batch {batch_idx}:")
        logging.debug(f"Box predictions shape: {box_preds.shape}")
        logging.debug(f"Class predictions shape: {cls_preds.shape}")
        logging.debug(f"Detection boxes shape: {detections['boxes'].shape}")
        logging.debug(f"Detection scores shape: {detections['scores'].shape}")
    except Exception as e:
        logging.error(f"Error logging outputs: {str(e)}")

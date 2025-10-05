import torch
import logging

def validate_tensor_shapes(predictions, targets):
    """Validate tensor shapes before loss computation"""
    try:
        if not isinstance(predictions, (list, tuple)):
            return False
        for pred in predictions:
            if pred.dim() != 4:  # [B, C, H, W]
                logging.error(f"Invalid prediction shape: {pred.shape}")
                return False
        return True
    except Exception as e:
        logging.error(f"Shape validation error: {str(e)}")
        return False

def reshape_predictions(pred, num_anchors):
    """Reshape predictions to correct dimensions"""
    B, C, H, W = pred.shape
    # Reshape [B, anchors*(5+classes), H, W] -> [B, H, W, anchors, 5+classes]
    return pred.view(B, num_anchors, -1, H, W).permute(0, 3, 4, 1, 2).contiguous()

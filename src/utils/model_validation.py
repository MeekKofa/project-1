import torch
import logging
from typing import List, Optional, Union, Tuple

def validate_model_io(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 448, 448),
    device: Optional[torch.device] = None
) -> bool:
    """Validate model input/output shapes and device compatibility"""
    try:
        if device is None:
            device = next(model.parameters()).device
            
        # Test input
        dummy_input = torch.randn(*input_shape, device=device)
        outputs = model(dummy_input)
        
        # Validate outputs
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]
            
        for out in outputs:
            if not isinstance(out, torch.Tensor):
                logging.error(f"Invalid output type: {type(out)}")
                return False
            if not out.device == device:
                logging.error(f"Output device mismatch: {out.device} vs {device}")
                return False
            if not torch.isfinite(out).all():
                logging.error("Non-finite values in output")
                return False
                
        return True
        
    except Exception as e:
        logging.error(f"Model validation error: {str(e)}")
        return False

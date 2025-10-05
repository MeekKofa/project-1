import torch
import logging
from typing import Tuple, Dict, Any

def validate_model_state(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 1
) -> Tuple[bool, str]:
    """Validate model state before training"""
    try:
        model.train()
        model_device = next(model.parameters()).device
        
        if str(model_device) != str(device):
            return False, f"Model device ({model_device}) doesn't match target device ({device})"
        
        # Create dummy batch with targets
        dummy_images = torch.randn(batch_size, 3, 448, 448).to(device)
        dummy_targets = [{
            'boxes': torch.tensor([[0.1, 0.1, 0.2, 0.2]], device=device),
            'labels': torch.tensor([0], device=device)
        } for _ in range(batch_size)]
        
        # Test forward pass
        with torch.no_grad():
            try:
                outputs = model(dummy_images, dummy_targets)
                if outputs is None:
                    return False, "Model returned None output"
                    
                # Verify outputs format
                if not isinstance(outputs, (tuple, list)):
                    return False, "Model output format invalid"
                    
                return True, None
                
            except Exception as e:
                return False, f"Forward pass failed: {str(e)}"
                
    except Exception as e:
        return False, f"Model state validation error: {str(e)}"

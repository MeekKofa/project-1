from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import logging

def check_model_output(outputs: List[torch.Tensor], detections: Dict) -> bool:
    """Validate model outputs during training"""
    try:
        # Basic format validation
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
            logging.warning("Invalid outputs structure")
            return False
            
        box_preds, cls_preds = outputs
        
        # Validate tensor types
        if not all(isinstance(t, torch.Tensor) for t in [box_preds, cls_preds]):
            logging.warning("Outputs must be PyTorch tensors")
            return False
            
        # Validate tensor shapes
        if len(box_preds.shape) != 3 or box_preds.shape[-1] != 4:
            logging.warning(f"Invalid box predictions shape: {box_preds.shape}")
            return False
            
        if len(cls_preds.shape) != 3:
            logging.warning(f"Invalid class predictions shape: {cls_preds.shape}")
            return False
            
        # Validate values
        if not (torch.isfinite(box_preds).all() and torch.isfinite(cls_preds).all()):
            logging.warning("Non-finite values in predictions")
            return False
            
        # Validate detections dictionary
        if not isinstance(detections, dict) or not all(k in detections for k in ['boxes', 'scores']):
            logging.warning("Invalid detections format")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Model output validation error: {str(e)}")
        return False

def validate_training_progress(
    loss: float,
    epoch: int,
    batch_idx: int,
    model: torch.nn.Module
) -> bool:
    """Validate training progress"""
    try:
        if not torch.isfinite(torch.tensor(loss)):
            return False
            
        # Check model gradients
        for param in model.parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    return False
                    
        return True
    except Exception as e:
        logging.error(f"Training progress validation error: {str(e)}")
        return False

def validate_input_targets(model: torch.nn.Module, images: torch.Tensor, targets: List[Dict]) -> Tuple[bool, Optional[str]]:
    """Validate model inputs and targets"""
    try:
        if images is None:
            return False, "Images cannot be None"
            
        if model.training and targets is None:
            return False, "Targets cannot be None during training"
            
        if not isinstance(images, torch.Tensor):
            return False, "Images must be a tensor"
            
        if model.training:
            if not isinstance(targets, (list, tuple)):
                return False, "Targets must be a list or tuple"
                
            if len(targets) != images.size(0):
                return False, "Batch size mismatch between images and targets"
                
        return True, None
        
    except Exception as e:
        return False, f"Input validation error: {str(e)}"

def validate_model_state(model: torch.nn.Module, device: torch.device, batch_size: int = 1) -> Tuple[bool, Optional[str]]:
    """Validate model state before training"""
    try:
        model.train()
        # Create dummy batch
        dummy_images = torch.randn(batch_size, 3, 448, 448).to(device)
        dummy_targets = [{
            'boxes': torch.tensor([[0.1, 0.1, 0.2, 0.2]], device=device),
            'labels': torch.tensor([0], device=device)
        } for _ in range(batch_size)]

        # Validate forward pass
        with torch.no_grad():
            outputs = model(dummy_images, dummy_targets)
            
            # Validate outputs
            if not isinstance(outputs, (tuple, list)):
                return False, "Invalid model output format"
            if not outputs or outputs[0] is None:
                return False, "Model returned None output"
            if not all(isinstance(out, torch.Tensor) for out in outputs):
                return False, "Invalid output tensor type"
                
            return True, None

    except Exception as e:
        return False, f"Model validation failed: {str(e)}"

def validate_model_training(model: torch.nn.Module, batch_size: int, device: torch.device) -> Tuple[bool, Optional[str]]:
    """Validate model training mode"""
    try:
        model.train()
        dummy_images = torch.randn(batch_size, 3, 448, 448).to(device)
        dummy_targets = [{
            'boxes': torch.tensor([[0.1, 0.1, 0.2, 0.2]], device=device),
            'labels': torch.tensor([0], device=device)
        } for _ in range(batch_size)]

        with torch.no_grad():
            outputs = model(dummy_images, dummy_targets)
            if not isinstance(outputs, (tuple, list)):
                return False, "Invalid model output format"
            if not outputs or outputs[0] is None:
                return False, "Model returned None output"
            return True, None

    except Exception as e:
        return False, f"Model validation failed: {str(e)}"
                
    except Exception as e:
        return False, f"Model state validation error: {str(e)}"

def validate_checkpoint_state(checkpoint: Dict, model: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
    """Validate checkpoint before loading"""
    try:
        if not all(k in checkpoint for k in ['model_state_dict', 'optimizer_state_dict', 'epoch', 'val_loss']):
            return False
            
        # Validate model state dict
        model_state = checkpoint['model_state_dict']
        if not isinstance(model_state, dict):
            return False
            
        # Verify model keys match
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(model_state.keys())
        if model_keys != checkpoint_keys:
            return False
            
        # Validate optimizer state
        optim_state = checkpoint['optimizer_state_dict']
        if not isinstance(optim_state, dict):
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Checkpoint validation error: {str(e)}")
        return False
        
    except Exception as e:
        logging.error(f"Checkpoint validation error: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Checkpoint validation error: {str(e)}")
        return False
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(model_state.keys())
        if model_keys != checkpoint_keys:
            return False
            
        # Validate optimizer state
        optim_state = checkpoint['optimizer_state_dict']
        if not isinstance(optim_state, dict):
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Checkpoint validation error: {str(e)}")
        return False
        
    except Exception as e:
        logging.error(f"Checkpoint validation error: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Checkpoint validation error: {str(e)}")
        return False
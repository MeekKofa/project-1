import os
import torch
import torch.nn as nn
import logging
import gc
from typing import Optional, Dict, Any, Tuple

def check_training_safety(device: torch.device) -> bool:
    """Check if it's safe to start/continue training"""
    try:
        if torch.cuda.is_available():
            # Clean memory first
            torch.cuda.empty_cache()
            gc.collect()
            
            # Get memory stats
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            # More lenient threshold - allow up to 75% memory usage
            if memory_allocated > total_memory * 0.75:
                logging.error("GPU memory usage too high")
                return False
                
            logging.info(f"GPU memory: {memory_allocated:.2f}GB used / {total_memory:.2f}GB total")
            return True
    except Exception as e:
        logging.error(f"Safety check failed: {str(e)}")
        return False
    return True

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Load and validate checkpoint"""
    try:
        if not os.path.exists(path):
            return False, None
            
        checkpoint = torch.load(path, map_location=device)
        # Validate checkpoint contents
        required_keys = {'epoch', 'model_state_dict', 'optimizer_state_dict', 'train_loss', 'val_loss'}
        if not all(k in checkpoint for k in required_keys):
            logging.error("Incomplete checkpoint")
            return False, None
            
        return True, checkpoint
    except Exception as e:
        logging.error(f"Checkpoint loading failed: {str(e)}")
        return False, None

def validate_checkpoint(checkpoint: Dict, model: nn.Module) -> bool:
    """Validate checkpoint before loading"""
    try:
        required_keys = {'model_state_dict', 'optimizer_state_dict', 'epoch', 'val_loss'}
        if not all(k in checkpoint for k in required_keys):
            return False
            
        model_state = checkpoint['model_state_dict']
        if not isinstance(model_state, dict):
            return False
            
        # Validate model state keys
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(model_state.keys())
        if not model_keys == checkpoint_keys:
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Checkpoint validation error: {str(e)}")
        return False

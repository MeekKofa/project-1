import torch
import logging

def validate_optimizer(optimizer, model):
    """Validate optimizer initialization"""
    try:
        # Check if optimizer has parameters
        if len(optimizer.param_groups) == 0:
            return False, "No parameter groups in optimizer"
            
        # Verify all model parameters are in optimizer
        model_params = set(model.parameters())
        optim_params = set()
        for group in optimizer.param_groups:
            optim_params.update(group['params'])
            
        if not model_params.issubset(optim_params):
            return False, "Not all model parameters are in optimizer"
            
        # Verify learning rate
        if optimizer.param_groups[0]['lr'] <= 0:
            return False, "Invalid learning rate"
            
        return True, "Optimizer validated successfully"
        
    except Exception as e:
        return False, f"Optimizer validation error: {str(e)}"

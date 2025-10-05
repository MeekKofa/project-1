import torch
import logging

def init_optimizer(model, lr, weight_decay):
    """Initialize and validate optimizer"""
    try:
        # Ensure model parameters are on correct device
        device = next(model.parameters()).device
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Verify optimizer state
        if len(optimizer.state_dict()['param_groups']) == 0:
            return None, "No parameters in optimizer"
            
        # Test optimizer step
        dummy_loss = sum(p.sum() * 0 for p in model.parameters())
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()
        
        return optimizer, None
        
    except Exception as e:
        return None, f"Optimizer initialization failed: {str(e)}"

def validate_optimizer_state(optimizer, model):
    """Validate optimizer state"""
    try:
        # Check optimizer parameters
        optimizer_params = set(p for group in optimizer.param_groups for p in group['params'])
        model_params = set(p for p in model.parameters() if p.requires_grad)
        
        if not model_params.issubset(optimizer_params):
            return False, "Not all model parameters are in optimizer"
            
        if any(group['lr'] <= 0 for group in optimizer.param_groups):
            return False, "Invalid learning rate"
            
        return True, None
        
    except Exception as e:
        return False, f"Optimizer validation error: {str(e)}"

import torch
import logging

def validate_model_init(model, device, batch_size=1):
    """Validate model initialization and forward pass"""
    try:
        # Test forward pass with dummy data
        dummy_image = torch.randn(batch_size, 3, 448, 448, device=device)
        dummy_target = [{
            'boxes': torch.tensor([[0.1, 0.1, 0.2, 0.2]], device=device),
            'labels': torch.ones(1, dtype=torch.long, device=device)
        } for _ in range(batch_size)]
        
        model.train()
        with torch.no_grad():
            outputs = model(dummy_image, dummy_target)
            
        if outputs is None:
            return False, "Model forward pass returned None"
            
        return True, None
    except Exception as e:
        return False, f"Model initialization validation failed: {str(e)}"

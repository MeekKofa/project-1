import torch
import logging

def validate_model_training_state(model, device, batch_size=1):
    """Validate model training state with dummy batch"""
    try:
        model.train()  # Ensure training mode
        
        # Create dummy batch
        dummy_images = torch.randn(batch_size, 3, 448, 448, device=device)
        dummy_boxes = torch.tensor([[0.1, 0.1, 0.2, 0.2]], device=device).repeat(batch_size, 1, 1)
        dummy_labels = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        dummy_targets = [
            {'boxes': boxes, 'labels': labels}
            for boxes, labels in zip(dummy_boxes, dummy_labels)
        ]
        
        # Test forward pass in training mode
        try:
            outputs = model(dummy_images, dummy_targets)
            if outputs is None:
                return False, "Model returned None output"
            return True, None
        except Exception as e:
            return False, f"Training forward pass failed: {str(e)}"
            
    except Exception as e:
        return False, f"Training validation error: {str(e)}"

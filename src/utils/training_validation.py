import torch
from typing import Tuple

def validate_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int
) -> Tuple[bool, str]:
    """Comprehensive training state validation"""
    try:
        # Check model
        model.train()
        if next(model.parameters()).device != device:
            return False, "Model not on correct device"

        # Validate with dummy data
        dummy_images = torch.randn(batch_size, 3, 448, 448, device=device)
        dummy_targets = [{
            'boxes': torch.tensor([[0.1, 0.1, 0.2, 0.2]], device=device),
            'labels': torch.tensor([0], device=device)
        } for _ in range(batch_size)]

        # Test forward pass and loss
        with torch.no_grad():
            try:
                outputs = model(dummy_images, dummy_targets)
                loss = model.compute_loss(outputs[0], dummy_targets)
                if not torch.isfinite(loss):
                    return False, "Model produced invalid loss"
            except Exception as e:
                return False, f"Forward pass failed: {str(e)}"

        # Check optimizer
        if len(optimizer.param_groups) == 0:
            return False, "Optimizer has no parameter groups"

        return True, None
    except Exception as e:
        return False, f"Training validation error: {str(e)}"

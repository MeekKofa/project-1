import torch
import logging
from typing import Tuple

def setup_model_parallel(model: torch.nn.Module, device: torch.device) -> Tuple[torch.nn.Module, bool]:
    """Setup model parallel state safely"""
    try:
        is_parallel = False
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            is_parallel = True
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
        return model.to(device), is_parallel
    except Exception as e:
        logging.error(f"Failed to setup parallel: {str(e)}")
        return model.to(device), False

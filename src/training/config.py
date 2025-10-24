"""
Enhanced training configuration with proper scheduling and monitoring.
"""

from typing import Dict, Any, Optional
import math
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import logging

logger = logging.getLogger(__name__)

class TrainingConfig:
    """
    Training configuration with best practices for object detection.
    Includes learning rate scheduling, gradient clipping, and early stopping.
    """
    
    def __init__(
        self,
        num_epochs: int = 100,
        batch_size: int = 16,
        base_lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        warmup_epochs: int = 5,
        patience: int = 10,
        scheduler_type: str = 'onecycle',
        grad_clip: float = 1.0,
        optimizer_type: str = 'sgd'
    ):
        """
        Initialize training configuration.
        
        Args:
            num_epochs: Maximum number of epochs
            batch_size: Batch size
            base_lr: Base learning rate
            momentum: SGD momentum
            weight_decay: Weight decay for regularization
            warmup_epochs: Number of warmup epochs
            patience: Early stopping patience
            scheduler_type: Learning rate scheduler ('onecycle' or 'cosine')
            grad_clip: Gradient clipping value
            optimizer_type: Optimizer type ('sgd' or 'adam')
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.scheduler_type = scheduler_type
        self.grad_clip = grad_clip
        self.optimizer_type = optimizer_type
        
        # Training state
        self.current_epoch = 0
        self.best_map50 = 0.0
        self.epochs_without_improvement = 0
        self.training_losses = []
        self.validation_metrics = []
        
    def create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Create optimizer with proper parameter groups."""
        # Separate backbone parameters for different learning rates
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
                
        param_groups = [
            {'params': backbone_params, 'lr': self.base_lr * 0.1},
            {'params': head_params, 'lr': self.base_lr}
        ]
        
        if self.optimizer_type == 'sgd':
            return SGD(
                param_groups,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            return Adam(
                param_groups,
                weight_decay=self.weight_decay
            )
            
    def create_scheduler(self, 
                        optimizer: torch.optim.Optimizer,
                        num_steps_per_epoch: int) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.scheduler_type == 'onecycle':
            return OneCycleLR(
                optimizer,
                max_lr=[self.base_lr * 0.1, self.base_lr],
                epochs=self.num_epochs,
                steps_per_epoch=num_steps_per_epoch,
                pct_start=self.warmup_epochs / self.num_epochs,
                anneal_strategy='cos'
            )
        else:
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.num_epochs // 3,  # First restart
                T_mult=2,  # Double period after each restart
                eta_min=1e-6
            )
            
    def update_state(self, 
                    epoch: int,
                    loss: float,
                    metrics: Dict[str, float]) -> bool:
        """
        Update training state and check for early stopping.
        
        Args:
            epoch: Current epoch number
            loss: Training loss
            metrics: Validation metrics
            
        Returns:
            bool: Whether to stop training
        """
        self.current_epoch = epoch
        self.training_losses.append(loss)
        self.validation_metrics.append(metrics)
        
        current_map50 = metrics.get('mAP50', 0.0)
        
        if current_map50 > self.best_map50:
            self.best_map50 = current_map50
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            
            # Check for early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                return True
                
            return False
            
    def get_current_lr(self, optimizer: torch.optim.Optimizer) -> float:
        """Get current learning rate."""
        return optimizer.param_groups[0]['lr']
        
    def clip_gradients(self, model: torch.nn.Module):
        """Apply gradient clipping."""
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.grad_clip
            )
            
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current training progress statistics."""
        return {
            'epoch': self.current_epoch,
            'best_map50': self.best_map50,
            'epochs_without_improvement': self.epochs_without_improvement,
            'training_losses': self.training_losses,
            'validation_metrics': self.validation_metrics
        }

def create_training_config(**kwargs) -> TrainingConfig:
    """Factory function to create training configuration."""
    return TrainingConfig(**kwargs)
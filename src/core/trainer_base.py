"""
Abstract base class for model trainers.

Provides a consistent interface for training different models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)


class TrainerBase(ABC):
    """
    Abstract base class for model trainers.

    Handles the training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            device: Device to train on
            config: Training configuration dict
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir or Path('./checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.best_epoch = 0

        # Move model to device
        self.model.to(self.device)

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics
        """
        pass

    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Training history dict
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model: {self.model.model_name}")
        logger.info(f"Device: {self.device}")

        history = {
            'train_loss': [],
            'val_metrics': []
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])

            # Validate
            val_metrics = self.validate()
            history['val_metrics'].append(val_metrics)

            # Log
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {train_metrics['loss']:.4f}, "
                f"mAP@0.5: {val_metrics.get('mAP@0.5', 0):.4f}"
            )

            # Save checkpoint
            self.save_checkpoint(val_metrics)

        return history

    def save_checkpoint(self, metrics: Dict[str, float]):
        """Save model checkpoint."""
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, latest_path)

        # Save best
        current_metric = metrics.get('mAP@0.5', 0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = self.current_epoch
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
            }, best_path)
            logger.info(f"Saved best model at epoch {self.current_epoch+1}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch+1}")

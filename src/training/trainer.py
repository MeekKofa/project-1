"""
Universal Trainer for Detection Models.

Works with any model registered in ModelRegistry.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import logging
from pathlib import Path
import time

from ..core.trainer_base import TrainerBase
from ..core.registry import ModelRegistry
from ..evaluation.metrics import compute_map

logger = logging.getLogger(__name__)


class DetectionTrainer(TrainerBase):
    """
    Universal trainer for detection models.

    Features:
    - Works with any model from ModelRegistry
    - Handles training loop, validation, checkpointing
    - Computes mAP metrics
    - Implements learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str,
        lr_scheduler: Optional[Any] = None,
        grad_clip: Optional[float] = None,
        log_interval: int = 10,
        val_interval: int = 1
    ):
        """
        Initialize trainer.

        Args:
            model: Detection model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            optimizer: Optimizer
            device: Device to train on
            output_dir: Directory to save checkpoints
            lr_scheduler: Optional learning rate scheduler
            grad_clip: Optional gradient clipping value
            log_interval: Log every N batches
            val_interval: Validate every N epochs
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            output_dir=output_dir
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.val_interval = val_interval

        # Move model to device
        self.model = self.model.to(self.device)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dict with training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]

            # Forward pass
            predictions = self.model(images, targets)

            # Compute loss
            loss = self.model.compute_loss(predictions, targets)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}")
                continue

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)

            # Update weights
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Logging
            if batch_idx % self.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        # Epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start

        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Update loss function epoch (for warmup)
        if hasattr(self.model, 'loss_fn'):
            self.model.loss_fn.current_epoch = epoch

        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Dict with validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                # Move data to device
                images = images.to(self.device)
                targets = [{k: v.to(self.device)
                            for k, v in t.items()} for t in targets]

                # Forward pass
                predictions = self.model(images, targets)

                # Compute loss
                loss = self.model.compute_loss(predictions, targets)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

                # Collect predictions for mAP
                box_preds, cls_preds, obj_preds = predictions
                all_predictions.append({
                    'boxes': box_preds,
                    'scores': torch.sigmoid(obj_preds.squeeze(-1)),
                    'labels': cls_preds.argmax(dim=-1)
                })
                all_targets.extend(targets)

        # Compute metrics
        avg_loss = total_loss / max(num_batches, 1)

        # Compute mAP
        try:
            map_score = compute_map(all_predictions, all_targets)
        except Exception as e:
            logger.error(f"Failed to compute mAP: {e}")
            map_score = 0.0

        metrics = {
            'val_loss': avg_loss,
            'mAP': map_score
        }

        logger.info(
            f"Epoch {epoch} Validation - "
            f"Loss: {avg_loss:.4f}, mAP: {map_score:.4f}"
        )

        return metrics

    def train(self, num_epochs: int, start_epoch: int = 0) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            num_epochs: Total number of epochs
            start_epoch: Starting epoch (for resuming)

        Returns:
            Dict with training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model: {self.model.__class__.__name__}")

        history = {
            'train_loss': [],
            'val_loss': [],
            'mAP': [],
            'best_epoch': 0,
            'best_mAP': 0.0
        }

        for epoch in range(start_epoch, num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(epoch)
            history['train_loss'].append(train_metrics['loss'])

            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Time: {train_metrics['epoch_time']:.2f}s, "
                f"LR: {train_metrics['lr']:.6f}"
            )

            # Validate
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate(epoch)
                history['val_loss'].append(val_metrics['val_loss'])
                history['mAP'].append(val_metrics['mAP'])

                # Check if best model
                if val_metrics['mAP'] > self.best_metric:
                    self.best_metric = val_metrics['mAP']
                    history['best_epoch'] = epoch
                    history['best_mAP'] = val_metrics['mAP']
                    self.save_checkpoint(
                        epoch=epoch,
                        metrics=val_metrics,
                        is_best=True
                    )
                    logger.info(
                        f"✓ New best model! mAP: {val_metrics['mAP']:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    metrics={'mAP': history['mAP'][-1]
                             if history['mAP'] else 0.0},
                    is_best=False
                )

        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed!")
        logger.info(
            f"Best mAP: {history['best_mAP']:.4f} at epoch {history['best_epoch']+1}")
        logger.info(f"{'='*60}\n")

        return history


def create_trainer(
    model_name: str,
    model_config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    grad_clip: Optional[float] = 10.0
) -> DetectionTrainer:
    """
    Create trainer with model from registry.

    Args:
        model_name: Name of model in registry (e.g., 'yolov8')
        model_config: Configuration for model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device to train on
        output_dir: Directory to save checkpoints
        learning_rate: Learning rate
        weight_decay: Weight decay
        grad_clip: Gradient clipping value

    Returns:
        Configured trainer
    """
    # Build model from registry
    model = ModelRegistry.build(model_name, **model_config)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=learning_rate * 0.01
    )

    # Create trainer
    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        lr_scheduler=lr_scheduler,
        grad_clip=grad_clip
    )

    return trainer

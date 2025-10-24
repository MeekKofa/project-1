"""
Enhanced training script with improved monitoring and validation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from tqdm import tqdm
import numpy as np

from ..models.simplified_detector import create_model
from ..training.config import create_training_config
from ..evaluation.metrics import DetectionMetricsSimple
from ..utils.data_validation import validate_targets

logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """
    Enhanced training manager with proper monitoring and validation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        output_dir: str,
        device: Optional[str] = None,
        patience: int = 15,  # Early stopping patience
        min_delta: float = 0.001  # Minimum change for improvement
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_config = create_training_config(**config)
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_map = 0.0
        self.patience_counter = 0
        self.best_model_path = self.output_dir / 'best_model.pth'
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_maps = []
        self.val_recalls = []
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = self.training_config.create_optimizer(self.model)
        self.scheduler = self.training_config.create_scheduler(
            self.optimizer,
            len(self.train_loader)
        )
        
        # Create metrics tracker
        self.metrics = DetectionMetricsSimple(num_classes=2)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.training_config.current_epoch + 1}") as pbar:
            for images, targets in pbar:
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Validate targets
                valid, message = validate_targets(targets)
                if not valid:
                    logger.warning(f"Skipping invalid batch: {message}")
                    continue
                
                # Forward pass
                loss_dict = self.model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                self.training_config.clip_gradients(self.model)
                
                # Update weights
                self.optimizer.step()
                
                # Update learning rate
                self.scheduler.step()
                
                # Update progress bar
                current_lr = self.training_config.get_current_lr(self.optimizer)
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
        return total_loss.item() / num_batches
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.metrics.reset()
        
        for images, targets in tqdm(self.val_loader, desc="Validating"):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Get predictions
            predictions = self.model(images)
            
            # Update metrics
            self.metrics.update(predictions, targets)
            
        # Compute metrics
        map50 = self.metrics.compute_map()
        precision = self.metrics.compute_precision()
        recall = self.metrics.compute_recall()
        
        return {
            'mAP50': map50,
            'precision': precision,
            'recall': recall
        }
        
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with validation metrics."""
        checkpoint = {
            'epoch': self.training_config.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_map': self.best_val_map,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maps': self.val_maps,
            'val_recalls': self.val_recalls
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.training_config.get_progress_stats()
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
    def train(self):
        """Run training loop with comprehensive validation."""
        logger.info("Starting training...")
        logger.info(f"Training on device: {self.device}")
        
        for epoch in range(self.training_config.num_epochs):
            # Train one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Run validation
            val_metrics = self.validate()
            current_map = val_metrics['mAP50']
            
            # Update validation metrics history
            self.val_maps.append(current_map)
            self.val_recalls.append(val_metrics['recall'])
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val mAP50: {current_map:.4f} | "
                f"Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f}"
            )
            
            # Check for improvement
            is_best = current_map > self.best_val_map + self.min_delta
            if is_best:
                self.best_val_map = current_map
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best=is_best)
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            is_best = val_metrics['mAP50'] > self.training_config.best_map50
            self.save_checkpoint(val_metrics, is_best)
            
            if should_stop:
                logger.info("Early stopping triggered")
                break
                
        # Save final training stats
        stats_path = self.output_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.training_config.get_progress_stats(), f, indent=2)
            
        logger.info("Training completed")
        logger.info(f"Best mAP50: {self.training_config.best_map50:.4f}")

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: str,
    device: Optional[str] = None
) -> nn.Module:
    """
    Train a model with the enhanced training pipeline.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        output_dir: Output directory for checkpoints
        device: Device to train on
        
    Returns:
        Trained model
    """
    # Create model
    model = create_model(num_classes=2)
    
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        device=device
    )
    
    # Run training
    trainer.train()
    
    return model
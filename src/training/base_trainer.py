"""
Base Trainer - Template for all model trainers.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image, ImageDraw
import re
from collections import defaultdict

from src.training.checkpoints import CheckpointManager


class BaseTrainer(ABC):
    """
    Base trainer class that all model trainers inherit from.
    Provides common training functionality and defines interface.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        output_dir: Path,
        logger: Any
    ):
        """
        Initialize base trainer.

        Args:
            model: Model to train
            config: Full configuration dictionary
            output_dir: Output directory for this training run
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger

        # Dataset & evaluation context
        self.dataset_info = config.get('dataset_info', {})
        self.class_names = self.dataset_info.get('class_names', [])
        self.num_classes = self.dataset_info.get('num_classes')

        validation_cfg = config.get('validation', {}) or {}
        evaluation_cfg = config.get('evaluation', {}) or {}
        self.validation_config = dict(validation_cfg)
        for key, value in evaluation_cfg.items():
            self.validation_config.setdefault(key, value)
        self.validation_config.setdefault('split', 'val')
        self.validation_config.setdefault('split_name', self.validation_config.get('split', 'val'))

        self.visualization_config = config.get('visualization', {})
        self.output_config = config.get('output', {})

        # Extract commonly used config values
        self.train_config = config.get('training', {})
        self.epochs = self.train_config.get('epochs', 100)
        self.batch_size = self.train_config.get('batch_size', 8)
        self.learning_rate = self.train_config.get('learning_rate', 0.001)
        self.device_config = config.get('device', {})

        # Setup device
        device_type = self.device_config.get('type', 'cuda')
        if device_type == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device_type == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        # Setup directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.metrics_dir = self.output_dir / 'metrics'
        self.viz_dir = self.output_dir / 'visualizations'

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            save_best=config.get('output', {}).get('save_best', True),
            save_latest=config.get('output', {}).get('save_latest', True),
            checkpoint_freq=self.train_config.get('checkpoint_freq', 10),
        )

        # Clean up stale per-epoch artifacts to avoid clutter
        self._cleanup_epoch_artifacts()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        self.epoch_history = []

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Mixed precision training
        self.use_mixed_precision = self.train_config.get(
            'mixed_precision', False)
        if self.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Using mixed precision training (FP16)")
        else:
            self.scaler = None


    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer based on config.

        Returns:
            Optimizer instance
        """
        optimizer_type = self.train_config.get('optimizer', 'adam').lower()
        weight_decay = self.train_config.get('weight_decay', 0.0001)

        optimizer_params = self.config.get(
            'optimizer_params', {}).get(optimizer_type, {})

        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
                **optimizer_params
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
                **optimizer_params
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
                **optimizer_params
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        self.logger.info(f"Created optimizer: {optimizer_type}")
        return optimizer

    def _create_scheduler(self) -> Optional[Any]:
        """
        Create learning rate scheduler based on config.

        Returns:
            Scheduler instance or None
        """
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine').lower()

        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config.get('gamma', 0.95)
            )
        else:
            self.logger.warning(
                f"Unknown scheduler type: {scheduler_type}, not using scheduler")
            return None

        self.logger.info(f"Created scheduler: {scheduler_type}")
        return scheduler

    @abstractmethod
    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch. Must be implemented by subclasses.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model. Must be implemented by subclasses.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            resume_from: Optional checkpoint path to resume from

        Returns:
            Training results dictionary
        """
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            checkpoint_info = self.checkpoint_manager.load_checkpoint(
                Path(resume_from),
                self.model,
                self.optimizer,
                self.scheduler,
                str(self.device)
            )
            start_epoch = checkpoint_info['epoch'] + 1
            self.logger.info(f"Resuming training from epoch {start_epoch}")

        # Training loop
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train one epoch
            self.logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            train_metrics = self.train_one_epoch(train_loader)

            # Validate
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Learning rate: {current_lr:.6f}")

            # Combine metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'lr': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
            }

            # Track metrics
            self.train_metrics.append(
                {**{'epoch': epoch + 1}, **train_metrics})
            if val_metrics:
                self.val_metrics.append(
                    {**{'epoch': epoch + 1}, **val_metrics})

            self.epoch_history.append(epoch_metrics)

            # Check if best model
            current_metric = val_metrics.get(
                'loss', train_metrics.get('loss', float('inf')))
            is_best = current_metric < self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.logger.info(
                    f"✓ New best model! Metric: {current_metric:.4f}")

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                epoch=epoch + 1,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=epoch_metrics,
                config=self.config,
                is_best=is_best
            )

            # Save metrics to CSV
            self._save_metrics_csv()

            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch time: {epoch_time:.2f}s")

        self.logger.info("\n" + "="*60)
        self.logger.info("Training completed!")
        self.logger.info(f"Best metric: {self.best_metric:.4f}")
        self.logger.info("="*60)

        return {
            'best_metric': self.best_metric,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }

    def _save_metrics_csv(self):
        """Save metrics to CSV files."""
        import csv

        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Save training metrics
        if self.train_metrics:
            train_csv = self.metrics_dir / 'train_metrics.csv'
            with open(train_csv, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=self.train_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.train_metrics)

        # Save validation metrics
        if self.val_metrics:
            val_split = self.validation_config.get('split_name', self.validation_config.get('split', 'val'))
            val_csv = self.metrics_dir / f'{val_split}_metrics.csv'
            with open(val_csv, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=self.val_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.val_metrics)

        # Save combined epoch metrics (train + val + lr)
        if self.epoch_history:
            summary_csv = self.metrics_dir / 'metrics_summary.csv'
            fieldnames = []
            keys = set()
            for row in self.epoch_history:
                keys.update(row.keys())

            if 'epoch' in keys:
                fieldnames.append('epoch')
                keys.remove('epoch')
            if 'lr' in keys:
                fieldnames.append('lr')
                keys.remove('lr')

            fieldnames.extend(sorted(keys))

            with open(summary_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.epoch_history)

        self._generate_metric_plots()

    def _generate_metric_plots(self) -> None:
        """Render training/validation metric visualizations."""
        try:
            from src.utils.metrics_plotter import MetricsPlotter
        except ImportError:
            self.logger.warning("matplotlib not available; skipping metric plots")
            return

        plotter = MetricsPlotter(self.metrics_dir / 'plots')
        plotter.update(self.train_metrics, self.val_metrics, self.epoch_history)

    def _cleanup_epoch_artifacts(self) -> None:
        """Remove legacy per-epoch JSON files so only consolidated outputs remain."""
        try:
            for json_file in self.metrics_dir.glob('val_metrics_epoch_*.json'):
                if json_file.is_file():
                    json_file.unlink()
        except FileNotFoundError:
            pass

        predictions_dir = self.output_dir / 'predictions'
        if predictions_dir.exists():
            for json_file in predictions_dir.glob('val_epoch_*.json'):
                if json_file.is_file():
                    json_file.unlink()

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _denormalize_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert normalized tensor image back to PIL image."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
        image = tensor * std + mean
        image = image.clamp(0, 1)
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def _draw_box(
        self,
        draw: ImageDraw.ImageDraw,
        box: List[float],
        label_idx: int,
        score: Optional[float],
        outline: str,
        width: int = 2,
    ):
        """Draw a bounding box with optional label text."""
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)

        label = self.class_names[label_idx] if self.class_names and label_idx < len(self.class_names) else f'class_{label_idx}'
        if score is not None:
            label = f"{label}: {score:.2f}"

        text_bbox = draw.textbbox((0, 0), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_origin = (max(x1, 0), max(y1 - text_height - 4, 0))
        padding = 2
        draw.rectangle(
            [
                text_origin,
                (text_origin[0] + text_width + 2 * padding,
                 text_origin[1] + text_height + 2 * padding),
            ],
            fill=outline,
        )
        draw.text(
            (text_origin[0] + padding, text_origin[1] + padding),
            label,
            fill='white'
        )

    def _save_visualizations(self, samples: List[Dict[str, Any]], split: str):
        """Persist visualizations for a subset of samples."""
        if not samples:
            return

        epoch_idx = self.current_epoch + 1
        total_epochs = max(self.epochs, epoch_idx)

        configured_epochs = self.visualization_config.get('epochs')
        if configured_epochs:
            if not isinstance(configured_epochs, (list, tuple, set)):
                configured_epochs = [configured_epochs]
            try:
                configured_epochs = {int(e) for e in configured_epochs}
            except (TypeError, ValueError):
                configured_epochs = set()
            if configured_epochs and epoch_idx not in configured_epochs:
                return
        else:
            save_interval = self.visualization_config.get('save_interval', 1) or 1
            try:
                save_interval = max(1, int(save_interval))
            except (TypeError, ValueError):
                save_interval = 1

            always_save_first = self.visualization_config.get('always_save_first', True)
            always_save_last = self.visualization_config.get('always_save_last', True)

            should_save = True
            if save_interval > 1 and (epoch_idx % save_interval != 0):
                should_save = False

            if not should_save and always_save_first and epoch_idx == 1:
                should_save = True
            if not should_save and always_save_last and epoch_idx == total_epochs:
                should_save = True

            if not should_save:
                return

        vis_dir = self.viz_dir / split
        vis_dir.mkdir(exist_ok=True)

        show_gt = self.visualization_config.get('show_ground_truth', True)
        show_preds = self.visualization_config.get('show_predictions', True)
        pred_conf_threshold = self.visualization_config.get('confidence_threshold', 0.5)

        for idx, sample in enumerate(samples):
            image = self._denormalize_image(sample['image'])
            draw = ImageDraw.Draw(image)

            if show_gt:
                gt = sample['target']
                boxes = gt['boxes']
                labels = gt['labels']
                for box, label in zip(boxes, labels):
                    self._draw_box(draw, box.tolist(), int(label), None, outline='green', width=3)

            if show_preds:
                preds = sample['pred']
                for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
                    if score < pred_conf_threshold:
                        continue
                    self._draw_box(draw, box.tolist(), int(label), float(score), outline='red', width=2)

            filename = vis_dir / f"{split}_epoch_{self.current_epoch + 1}_img_{sample['image_id']}_{idx}.png"
            image.save(filename)

        max_epochs_to_keep = self.visualization_config.get('max_epochs_to_keep')
        if max_epochs_to_keep:
            try:
                max_epochs_to_keep = int(max_epochs_to_keep)
            except (TypeError, ValueError):
                max_epochs_to_keep = 0

            if max_epochs_to_keep > 0:
                self._prune_visualization_outputs(vis_dir, split, max_epochs_to_keep)

    def _prune_visualization_outputs(self, vis_dir: Path, split: str, max_epochs: int) -> None:
        """Remove visualization outputs beyond the configured limit."""
        pattern = re.compile(rf"{split}_epoch_(\\d+)_")
        epoch_to_paths: Dict[int, List[Path]] = defaultdict(list)

        for image_path in vis_dir.glob(f"{split}_epoch_*_img_*.png"):
            match = pattern.search(image_path.name)
            if not match:
                continue
            try:
                epoch_val = int(match.group(1))
            except ValueError:
                continue
            epoch_to_paths[epoch_val].append(image_path)

        if len(epoch_to_paths) <= max_epochs:
            return

        for epoch_val in sorted(epoch_to_paths.keys())[:-max_epochs]:
            for image_path in epoch_to_paths[epoch_val]:
                try:
                    image_path.unlink()
                except FileNotFoundError:
                    pass

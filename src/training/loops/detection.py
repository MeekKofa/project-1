"""
Detection Training Loop - Generic detection model trainer.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Dict, List
from collections import defaultdict
from tqdm import tqdm

from src.training.base_trainer import BaseTrainer
from src.training.evaluation import (
    DetectionMetricAccumulator,
    postprocess_batch_predictions,
)


class DetectionTrainer(BaseTrainer):
    """
    Generic trainer for object detection models.
    Works with Faster R-CNN, YOLOv8, and other detection models.
    """

    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        log_freq = self.train_config.get('log_freq', 10)
        component_totals: Dict[str, float] = defaultdict(float)

        # Update loss functions that track epoch state (e.g., YOLO warmup)
        if hasattr(self.model, 'loss_fn') and hasattr(self.model.loss_fn, 'current_epoch'):
            self.model.loss_fn.current_epoch = self.current_epoch

        # Progress bar
        pbar = tqdm(
            train_loader, desc=f"Training Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images, targets = batch
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]

            # Determine input format expected by model
            model_input_type = getattr(self.model, 'input_type', 'list')
            if model_input_type == 'tensor':
                images_for_model = torch.stack(images, dim=0)
            else:
                images_for_model = images

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            if self.scaler:
                with torch.cuda.amp.autocast():
                    raw_losses = self.model(images_for_model, targets)
                    if isinstance(raw_losses, dict):
                        loss_dict = raw_losses
                    elif isinstance(raw_losses, torch.Tensor):
                        loss_dict = {'loss': raw_losses}
                    else:
                        raise TypeError(
                            f"Unexpected loss output type: {type(raw_losses)}")
                    losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                self.scaler.scale(losses).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass
                raw_losses = self.model(images_for_model, targets)
                if isinstance(raw_losses, dict):
                    loss_dict = raw_losses
                elif isinstance(raw_losses, torch.Tensor):
                    loss_dict = {'loss': raw_losses}
                else:
                    raise TypeError(
                        f"Unexpected loss output type: {type(raw_losses)}")
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                losses.backward()
                self.optimizer.step()

            for name, value in loss_dict.items():
                component_totals[name] += float(value.item())

            # Track metrics
            total_loss += losses.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Log periodically
            if (batch_idx + 1) % log_freq == 0:
                self.logger.info(
                    f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                    f"Loss: {avg_loss:.4f}"
                )

        # Calculate epoch metrics
        epoch_metrics = {
            'loss': total_loss / num_batches,
        }

        for name, value in component_totals.items():
            if name == 'loss':
                continue
            epoch_metrics[name] = value / num_batches

        self.logger.info(f"Training - Loss: {epoch_metrics['loss']:.4f}")

        return epoch_metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        component_totals: Dict[str, float] = defaultdict(float)

        validation_cfg = self.validation_config
        conf_threshold = validation_cfg.get('conf_threshold', 0.25)
        metric_iou = validation_cfg.get('iou_threshold', 0.5)
        max_det = validation_cfg.get('max_det', 300)
        split_name = validation_cfg.get('split', validation_cfg.get('split_name', 'val'))

        model_cfg = self.config.get('models', {})
        model_name = self.config.get('model')
        current_model_cfg = model_cfg.get(model_name, {})
        if not current_model_cfg and 'yolov8' in model_cfg:
            current_model_cfg = model_cfg['yolov8']
        model_eval_cfg = current_model_cfg.get('config', {})
        nms_iou = model_eval_cfg.get('iou_threshold', metric_iou)

        num_classes = self.num_classes or getattr(self.model, 'num_classes', None)
        if num_classes is None or num_classes == 0:
            num_classes = len(self.class_names) if self.class_names else 1

        accumulator = DetectionMetricAccumulator(num_classes=num_classes, iou_threshold=metric_iou)

        vis_enabled = self.visualization_config.get('enabled', False) and self.output_config.get('save_visualizations', True)
        max_vis = self.visualization_config.get('num_samples', 8)
        visual_samples: List[Dict[str, Any]] = []

        # Keep loss functions informed about epoch for warmup behaviour
        if hasattr(self.model, 'loss_fn') and hasattr(self.model.loss_fn, 'current_epoch'):
            self.model.loss_fn.current_epoch = self.current_epoch

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")

            for batch in pbar:
                # Move batch to device
                images, targets = batch
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device)
                            for k, v in t.items()} for t in targets]

                image_shapes = [(
                    int(img.shape[-2]),
                    int(img.shape[-1])
                ) for img in images]

                model_input_type = getattr(self.model, 'input_type', 'list')
                if model_input_type == 'tensor':
                    images_for_model = torch.stack(images, dim=0)
                else:
                    images_for_model = images

                # Gather predictions for metrics (model in eval mode)
                self.model.eval()
                raw_predictions = self.model(images_for_model)
                processed_predictions = postprocess_batch_predictions(
                    raw_predictions,
                    image_shapes=image_shapes,
                    conf_threshold=conf_threshold,
                    nms_iou=nms_iou,
                    max_det=max_det,
                )

                processed_cpu = [
                    {
                        'boxes': det['boxes'].detach().cpu(),
                        'scores': det['scores'].detach().cpu(),
                        'labels': det['labels'].detach().cpu(),
                    }
                    for det in processed_predictions
                ]

                targets_cpu = [
                    {k: v.detach().cpu() for k, v in t.items()}
                    for t in targets
                ]

                accumulator.update(processed_cpu, targets_cpu)

                if vis_enabled and len(visual_samples) < max_vis:
                    for img_tensor, pred_cpu, target_cpu in zip(images, processed_cpu, targets_cpu):
                        if len(visual_samples) >= max_vis:
                            break

                        image_id_tensor = target_cpu.get('image_id')
                        if image_id_tensor is not None:
                            if isinstance(image_id_tensor, torch.Tensor):
                                image_id = int(image_id_tensor.flatten()[0].item())
                            else:
                                image_id = int(image_id_tensor)
                        else:
                            # Generate a sequential ID if none provided
                            image_id = len(visual_samples)
                            self.logger.warning(
                                "No image_id found in target during validation. "
                                "Using sequential ID %d. This may affect visualization ordering.",
                                image_id
                            )

                        visual_samples.append({
                            'image': img_tensor.detach().cpu(),
                            'pred': pred_cpu,
                            'target': target_cpu,
                            'image_id': image_id,
                        })

                # Forward pass
                self.model.train()  # Some models need train mode for loss calculation
                raw_losses = self.model(images_for_model, targets)
                if isinstance(raw_losses, dict):
                    loss_dict = raw_losses
                elif isinstance(raw_losses, torch.Tensor):
                    loss_dict = {'loss': raw_losses}
                else:
                    raise TypeError(
                        f"Unexpected loss output type: {type(raw_losses)}")
                losses = sum(loss for loss in loss_dict.values())

                # Track metrics
                total_loss += losses.item()
                num_batches += 1

                for name, value in loss_dict.items():
                    component_totals[name] += float(value.item())

                self.model.eval()

                # Update progress bar
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Calculate validation metrics
        base_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics_summary = accumulator.compute()

        val_metrics = {
            'loss': base_loss,
            'map_50': metrics_summary['map_50'],
            'precision_50': metrics_summary['precision_50'],
            'recall_50': metrics_summary['recall_50'],
            'f1_50': metrics_summary['f1_50'],
        }

        for name, value in component_totals.items():
            if name == 'loss':
                continue
            val_metrics[name] = value / num_batches

        log_message = (
            f"Validation - Loss: {val_metrics['loss']:.4f}, "
            f"mAP@0.5: {val_metrics['map_50']:.4f}, "
            f"Precision@0.5: {val_metrics['precision_50']:.4f}, "
            f"Recall@0.5: {val_metrics['recall_50']:.4f}"
        )
        self.logger.info(log_message)

        # Persist evaluation artifacts
        metrics_summary['epoch'] = self.current_epoch + 1
        metrics_summary['split'] = split_name
        self._save_consolidated_metrics(metrics_summary, split=split_name)

        if self.output_config.get('save_predictions', True):
            self._save_consolidated_predictions(
                accumulator,
                epoch=self.current_epoch + 1,
                metadata={
                    'conf_threshold': conf_threshold,
                    'nms_iou': nms_iou,
                    'max_detections': max_det,
                    'split': split_name,
                },
                split=split_name,
            )

        if vis_enabled:
            self._save_visualizations(visual_samples, split=split_name)

        return val_metrics

    def _save_consolidated_metrics(self, metrics_summary: Dict[str, Any], split: str) -> None:
        """Persist validation metrics into a single JSON file with history."""
        if self.class_names:
            for entry in metrics_summary.get('per_class', []):
                class_idx = entry.get('class_id')
                if class_idx is not None and class_idx < len(self.class_names):
                    entry['class_name'] = self.class_names[class_idx]

        metrics_path = self.metrics_dir / f'{split}_metrics.json'
        history: List[Dict[str, Any]]

        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as fp:
                    existing = json.load(fp)
                history = existing.get('history', [])
            except (json.JSONDecodeError, OSError):
                history = []
        else:
            history = []

        history = [entry for entry in history if entry.get('epoch') != metrics_summary.get('epoch')]
        history.append(metrics_summary)
        history.sort(key=lambda item: item.get('epoch', 0))

        payload = {
            'latest': metrics_summary,
            'history': history,
        }

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as fp:
            json.dump(payload, fp, indent=2)

    def _save_consolidated_predictions(
        self,
        accumulator: DetectionMetricAccumulator,
        epoch: int,
        metadata: Dict[str, Any],
        split: str,
    ) -> None:
        """Write predictions/targets from the most recent validation step to a single file."""
        predictions_path = self.output_dir / 'predictions' / f'{split}_predictions.json'
        predictions_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'epoch': epoch,
            'metadata': metadata,
            'predictions': accumulator.sample_predictions,
            'targets': accumulator.sample_targets,
        }

        with open(predictions_path, 'w') as fp:
            json.dump(payload, fp)

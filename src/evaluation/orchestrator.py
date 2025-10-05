"""Evaluation orchestrator for running inference on saved checkpoints."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.config.manager import ConfigManager
from src.loaders.registry import get_dataset, get_dataset_info
from src.loaders.transforms import detection_collate_fn
from src.models.registry import get_model
from src.training.loops.detection import DetectionTrainer


class EvaluationOrchestrator:
    """Coordinate end-to-end evaluation for a trained checkpoint."""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: Path,
        split: str = 'test',
        batch_size: Optional[int] = None,
        run_name: Optional[str] = None,
    ) -> None:
        self.original_config = config
        self.config = deepcopy(config)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model_name = self.config.get('model')
        self.dataset_name = self.config.get('dataset')
        if not self.model_name:
            raise ValueError("Model name must be provided in configuration for evaluation")
        if not self.dataset_name:
            raise ValueError("Dataset name must be provided in configuration for evaluation")

        self.split = split
        self.batch_size = batch_size or self.config.get('evaluation', {}).get('batch_size', 8)
        self.batch_size = int(self.batch_size)

        base_output = Path(self.config.get('output', {}).get('base_dir', 'outputs'))
        run_name = run_name or self._derive_run_name()
        self.output_dir = base_output / self.dataset_name / self.model_name / 'evaluations' / run_name

        # Create directory scaffold
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / 'metrics'
        self.predictions_dir = self.output_dir / 'predictions'
        self.logs_dir = self.output_dir / 'logs'
        self.visualizations_dir = self.output_dir / 'visualizations'

        for directory in [self.metrics_dir, self.predictions_dir, self.logs_dir, self.visualizations_dir]:
            directory.mkdir(exist_ok=True)
        (self.visualizations_dir / split).mkdir(exist_ok=True)

        # Prepare dataset information
        dataset_info = get_dataset_info(self.dataset_name)
        if dataset_info:
            self.config.setdefault('dataset_info', {}).update(dataset_info)

        self.config.setdefault('validation', {})
        self.config['validation']['split'] = split
        self.config['validation']['split_name'] = split

        # Reduce checkpoint behaviour during evaluation
        self.config.setdefault('output', {})
        self.config['output'].setdefault('save_best', False)
        self.config['output'].setdefault('save_latest', False)
        self.config['output'].setdefault('save_visualizations', True)

        evaluation_cfg = self.config.setdefault('evaluation', {})
        evaluation_cfg['split'] = split
        evaluation_cfg['batch_size'] = self.batch_size
        if 'save_predictions' in evaluation_cfg:
            self.config['output']['save_predictions'] = evaluation_cfg['save_predictions']
        else:
            self.config['output'].setdefault('save_predictions', True)

        # Save merged config for traceability
        self._save_config_snapshot()

        self.logger = self._setup_logger(run_name)
        self.logger.info("=" * 60)
        self.logger.info("Evaluation Orchestrator Initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Split: {self.split}")
        self.logger.info(f"Checkpoint: {self.checkpoint_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on the configured checkpoint."""
        data_loader = self._create_dataloader()
        trainer = self._create_trainer()

        checkpoint_info = self._load_checkpoint(trainer.model, trainer.device)
        checkpoint_epoch = checkpoint_info.get('epoch') or 1
        trainer.current_epoch = max(checkpoint_epoch - 1, 0)
        trainer.epochs = max(checkpoint_epoch, 1)
        trainer.validation_config['split'] = self.split
        trainer.validation_config['split_name'] = self.split

        self.logger.info("Starting evaluation...")
        metrics = trainer.validate(data_loader)

        eval_epoch_idx = trainer.current_epoch + 1
        trainer.val_metrics = [{'epoch': eval_epoch_idx, **metrics}]
        trainer.epoch_history = [{
            'epoch': eval_epoch_idx,
            'lr': 0.0,
            **{f'val_{k}': v for k, v in metrics.items()}
        }]

        trainer._save_metrics_csv()

        summary_path = self._save_summary(metrics, checkpoint_info)

        results = {
            'metrics': metrics,
            'summary_path': summary_path,
            'output_dir': self.output_dir,
            'checkpoint': checkpoint_info,
        }

        self.logger.info("Evaluation complete!")
        self.logger.info(f"Primary metric (mAP@0.5): {metrics.get('map_50', 0.0):.4f}")
        self.logger.info(f"Results written to: {summary_path}")

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_run_name(self) -> str:
        stem = self.checkpoint_path.stem
        if stem in {'best', 'latest'}:
            parent = self.checkpoint_path.parent.name
            stem = f"{parent}_{stem}"
        return f"{stem}_{self.split}"

    def _setup_logger(self, run_name: str) -> logging.Logger:
        log_file = self.logs_dir / 'eval.log'
        logger = logging.getLogger(f"{self.dataset_name}_{self.model_name}_eval_{run_name}")
        logger.setLevel(logging.INFO)
        logger.handlers = []

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        log_format = self.config.get('logging', {}).get(
            'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def _create_dataloader(self) -> DataLoader:
        dataset = get_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            config=self.config
        )

        num_workers = self.config.get('training', {}).get('num_workers', 4)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=detection_collate_fn
        )

        self.logger.info(f"Dataset size: {len(dataset)} samples")
        self.logger.info(f"Batches: {len(loader)}")
        return loader

    def _create_trainer(self) -> DetectionTrainer:
        model = get_model(
            model_name=self.model_name,
            num_classes=self.config.get('dataset_info', {}).get('num_classes'),
            config=self.config
        )

        trainer = DetectionTrainer(
            model=model,
            config=self.config,
            output_dir=self.output_dir,
            logger=self.logger
        )

        # Avoid saving checkpoints during evaluation
        trainer.checkpoint_manager.save_best = False
        trainer.checkpoint_manager.save_latest = False
        trainer.checkpoint_manager.checkpoint_freq = 0

        return trainer

    def _load_checkpoint(self, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
        checkpoint = torch.load(self.checkpoint_path, map_location=device)

        epoch = checkpoint.get('epoch') if isinstance(checkpoint, dict) else None
        metrics = checkpoint.get('metrics') if isinstance(checkpoint, dict) else {}

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        info = {
            'epoch': epoch,
            'metrics': metrics,
        }
        if epoch is not None:
            self.logger.info(f"Loaded checkpoint trained up to epoch {epoch}")
        return info

    def _save_summary(self, metrics: Dict[str, Any], checkpoint_info: Dict[str, Any]) -> Path:
        summary_path = self.metrics_dir / f'{self.split}_results.txt'
        predictions_path = self.predictions_dir / f'{self.split}_predictions.json'
        metrics_json = self.metrics_dir / f'{self.split}_metrics.json'
        metrics_csv = self.metrics_dir / f'{self.split}_metrics.csv'

        with open(summary_path, 'w') as fp:
            fp.write("=" * 60 + "\n")
            fp.write("EVALUATION RESULTS\n")
            fp.write("=" * 60 + "\n\n")
            fp.write(f"Model: {self.model_name}\n")
            fp.write(f"Dataset: {self.dataset_name}\n")
            fp.write(f"Split: {self.split}\n")
            fp.write(f"Checkpoint: {self.checkpoint_path}\n")
            if checkpoint_info.get('epoch') is not None:
                fp.write(f"Trained Epoch: {checkpoint_info['epoch']}\n")
            fp.write(f"Output directory: {self.output_dir}\n\n")

            fp.write("Metrics:\n")
            for key in sorted(metrics.keys()):
                value = metrics[key]
                if isinstance(value, (float, int)):
                    fp.write(f"  {key}: {value:.4f}\n")
            fp.write("\n")

            fp.write("Artifacts:\n")
            fp.write(f"  Metrics JSON: {metrics_json}\n")
            fp.write(f"  Metrics CSV: {metrics_csv}\n")
            fp.write(f"  Summary CSV: {self.metrics_dir / 'metrics_summary.csv'}\n")
            fp.write(f"  Predictions: {predictions_path}\n")
            fp.write(f"  Visualizations: {self.visualizations_dir / self.split}\n")
            fp.write("\n" + "=" * 60 + "\n")

        return summary_path

    def _save_config_snapshot(self) -> None:
        config_path = self.output_dir / 'config_used.yaml'
        manager = ConfigManager(self.original_config.get('_metadata', {}).get('config_file'))
        manager.save_config(self.config, config_path)


def evaluate_model(
    config: Dict[str, Any],
    checkpoint_path: str,
    split: str = 'test',
    batch_size: Optional[int] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    orchestrator = EvaluationOrchestrator(
        config=config,
        checkpoint_path=Path(checkpoint_path),
        split=split,
        batch_size=batch_size,
        run_name=run_name,
    )
    return orchestrator.evaluate()

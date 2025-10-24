"""
Training Orchestrator - Main entry point for all training.
Coordinates model loading, dataset loading, and training execution.
"""

from pathlib import Path
from typing import Dict, Any
import logging

from src.models.registry import get_model, get_model_info
from src.loaders.registry import get_dataset, get_dataset_info
from src.config.manager import ConfigManager


class TrainingOrchestrator:
    """
    Orchestrates the entire training process.
    Single entry point that coordinates all components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training orchestrator.

        Args:
            config: Full merged configuration dictionary
        """
        self.config = config
        self.model_name = config.get('model')
        self.dataset_name = config.get('dataset')

        if not self.model_name:
            raise ValueError("Model name not specified in config")
        if not self.dataset_name:
            raise ValueError("Dataset name not specified in config")

        # Setup output directory structure
        self.output_dir = Path(config.get(
            'output', {}).get('base_dir', 'outputs'))
        self.output_dir = self.output_dir / self.dataset_name / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.metrics_dir = self.output_dir / 'metrics'
        self.viz_dir = self.output_dir / 'visualizations'

        for directory in [self.checkpoint_dir, self.log_dir, self.metrics_dir, self.viz_dir]:
            directory.mkdir(exist_ok=True)

        # Create visualization subdirectories
        (self.viz_dir / 'training').mkdir(exist_ok=True)
        (self.viz_dir / 'evaluation').mkdir(exist_ok=True)
        (self.viz_dir / 'predictions').mkdir(exist_ok=True)

        # Setup logging
        self.logger = self._setup_logger()

        self.logger.info("="*60)
        self.logger.info("Training Orchestrator Initialized")
        self.logger.info("="*60)
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("="*60)

        # Get dataset info (with auto-detection)
        self.dataset_info = get_dataset_info(self.dataset_name)
        num_classes = self.dataset_info.get('num_classes')
        class_names = self.dataset_info.get('class_names', [])

        # Expose dataset info to downstream components via config
        dataset_info_cfg = self.config.setdefault('dataset_info', {})
        if num_classes is not None:
            dataset_info_cfg['num_classes'] = num_classes
        if class_names:
            dataset_info_cfg['class_names'] = class_names
        dataset_info_cfg.setdefault('dataset', self.dataset_name)

        self.logger.info(f"Dataset info: {num_classes} classes")
        if class_names:
            self.logger.info(f"Class names: {class_names}")

        # Load model from registry
        self.logger.info(f"\nLoading model: {self.model_name}")
        self.model = get_model(
            model_name=self.model_name,
            num_classes=num_classes,
            config=self.config
        )

        # Save merged config for reproducibility
        config_manager = ConfigManager()
        config_save_path = self.output_dir / 'config_used.yaml'
        config_manager.save_config(self.config, config_save_path)

        self.logger.info(f"\n✓ Orchestrator ready!")

    def _setup_logger(self) -> logging.Logger:
        """
        Setup model-specific logger.

        Returns:
            Configured logger instance
        """
        log_file = self.log_dir / 'train.log'

        # Create logger
        logger = logging.getLogger(f'{self.dataset_name}_{self.model_name}')
        logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        log_format = self.config.get('logging', {}).get(
            'format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def create_dataloaders(self):
        """
        Create train and validation dataloaders.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        from torch.utils.data import DataLoader
        from src.loaders.transforms import detection_collate_fn

        self.logger.info("\nCreating dataloaders...")

        # Get datasets
        train_dataset = get_dataset(
            dataset_name=self.dataset_name,
            split='train',
            config=self.config
        )

        val_dataset = get_dataset(
            dataset_name=self.dataset_name,
            split='val',
            config=self.config
        )

        # Create dataloaders
        train_config = self.config.get('training', {})
        batch_size = train_config.get('batch_size', 8)
        num_workers = train_config.get('num_workers', 4)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=detection_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=detection_collate_fn
        )

        self.logger.info(f"✓ Train batches: {len(train_loader)}")
        self.logger.info(f"✓ Val batches: {len(val_loader)}")

        return train_loader, val_loader

    def create_trainer(self):
        """
        Create appropriate trainer for the model.

        Returns:
            Trainer instance
        """
        from src.training.base_trainer import BaseTrainer

        # Import detection trainer for all object detection models
        from src.training.loops.detection import DetectionTrainer

        # Determine trainer type based on model
        if 'faster_rcnn' in self.model_name:
            # Use DetectionTrainer for Faster R-CNN
            trainer = DetectionTrainer(
                model=self.model,
                config=self.config,
                output_dir=self.output_dir,
                logger=self.logger
            )
        elif 'yolov8' in self.model_name or self.model_name == 'vgg16_yolov8':
            # Use DetectionTrainer for YOLOv8-based models (including VGG16-YOLOv8)
            trainer = DetectionTrainer(
                model=self.model,
                config=self.config,
                output_dir=self.output_dir,
                logger=self.logger
            )
        else:
            # Default detection trainer
            from src.training.loops.detection import DetectionTrainer
            trainer = DetectionTrainer(
                model=self.model,
                config=self.config,
                output_dir=self.output_dir,
                logger=self.logger
            )

        return trainer

    def train(self):
        """
        Execute the complete training process.

        Returns:
            Training results dictionary
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("Starting Training Process")
        self.logger.info("="*60)

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()

        # Create trainer
        self.logger.info("\nCreating trainer...")
        trainer = self.create_trainer()

        # Check for resume
        resume_from = self.config.get('resume_checkpoint')

        # Train
        self.logger.info("\nBeginning training...")
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from=resume_from
        )

        # Save final results
        self._save_final_results(results)

        self.logger.info("\n" + "="*60)
        self.logger.info("Training Process Complete!")
        self.logger.info("="*60)
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"Best metric: {results['best_metric']:.4f}")

        return results

    def _save_final_results(self, results: Dict[str, Any]):
        """
        Save final training results to text file.

        Args:
            results: Training results dictionary
        """
        results_file = self.metrics_dir / 'eval_results.txt'

        with open(results_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TRAINING RESULTS\n")
            f.write("="*60 + "\n\n")

            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")

            f.write(f"Best metric: {results['best_metric']:.4f}\n")
            f.write(
                f"Total epochs: {self.config.get('training', {}).get('epochs', 0)}\n\n")

            # Final training metrics
            if results.get('train_metrics'):
                final_train = results['train_metrics'][-1]
                f.write("Final Training Metrics:\n")
                for key, value in final_train.items():
                    if key != 'epoch':
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")

            # Final validation metrics
            if results.get('val_metrics'):
                final_val = results['val_metrics'][-1]
                f.write("Final Validation Metrics:\n")
                for key, value in final_val.items():
                    if key != 'epoch':
                        f.write(f"  {key}: {value:.4f}\n")

            f.write("\n" + "="*60 + "\n")

        self.logger.info(f"✓ Saved results to: {results_file}")


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to train a model.

    Args:
        config: Full configuration dictionary

    Returns:
        Training results
    """
    orchestrator = TrainingOrchestrator(config)
    return orchestrator.train()

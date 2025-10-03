"""
Universal Training Configuration System.

Zero hardcoding - everything is configurable via arguments or config files.
"""

import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingConfig:
    """
    Universal training configuration with zero hardcoding.

    Supports:
    - Command-line arguments
    - YAML config files
    - JSON config files
    - Programmatic configuration
    """

    # Default values (can be overridden)
    DEFAULTS = {
        # Model configuration
        'model': 'yolov8',
        'num_classes': None,  # Auto-detect from dataset
        'pretrained': False,
        'freeze_backbone': False,

        # Dataset configuration
        'dataset_name': None,
        'dataset_root': None,
        'train_images': None,
        'train_labels': None,
        'val_images': None,
        'val_labels': None,
        'test_images': None,
        'test_labels': None,
        'image_size': 640,
        'label_format': 'yolo',  # 'yolo', 'coco', 'voc'

        # Training hyperparameters
        'epochs': 100,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'momentum': 0.9,

        # Optimizer configuration
        'optimizer': 'adamw',  # 'sgd', 'adam', 'adamw', 'rmsprop'
        'optimizer_params': {},

        # Scheduler configuration
        'scheduler': 'cosine',  # 'step', 'cosine', 'plateau', 'onecycle', None
        'scheduler_params': {},

        # Loss weights
        'box_weight': 7.5,
        'cls_weight': 0.5,
        'obj_weight': 1.0,

        # Augmentation
        'augment': True,
        'augment_params': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.0,
            'rotation': 10,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
        },

        # Regularization
        'dropout': 0.3,
        'label_smoothing': 0.0,
        'mixup_alpha': 0.0,
        'cutmix_alpha': 0.0,

        # Training strategy
        'warmup_epochs': 5,
        'early_stopping': True,
        'early_stopping_patience': 20,
        'gradient_clip': 1.0,
        'mixed_precision': True,

        # Validation
        'val_interval': 1,
        'val_metric': 'mAP',  # 'mAP', 'loss', 'f1'

        # Checkpointing
        'save_interval': 5,
        'save_best_only': False,
        'checkpoint_dir': None,

        # Output configuration
        'output_dir': 'outputs',
        'experiment_name': None,
        'log_interval': 10,
        'save_predictions': True,

        # Device configuration
        'device': 'cuda',
        'num_workers': 4,
        'pin_memory': True,

        # Debugging
        'debug': False,
        'profile': False,
        'seed': 42,
    }

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Optional dictionary with configuration values
        """
        # Start with defaults
        self.config = self.DEFAULTS.copy()

        # Update with provided config
        if config_dict is not None:
            self.update(config_dict)

    def update(self, config_dict: Dict[str, Any]):
        """
        Update configuration with new values.

        Args:
            config_dict: Dictionary with configuration updates
        """
        for key, value in config_dict.items():
            if key in self.config:
                self.config[key] = value
            else:
                logger.warning(f"Unknown configuration key: {key}")

    def __getitem__(self, key: str) -> Any:
        """Get configuration value."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return key in self.config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config.copy()

    def save(self, path: str):
        """
        Save configuration to file.

        Args:
            path: Path to save configuration (YAML or JSON)
        """
        path = Path(path)

        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def from_file(cls, path: str) -> 'TrainingConfig':
        """
        Load configuration from file.

        Args:
            path: Path to configuration file (YAML or JSON)

        Returns:
            TrainingConfig instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info(f"Configuration loaded from {path}")
        return cls(config_dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """
        Create configuration from command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            TrainingConfig instance
        """
        # Convert args to dict, excluding None values
        config_dict = {k: v for k, v in vars(args).items() if v is not None}
        return cls(config_dict)

    def validate(self):
        """Validate configuration."""
        errors = []

        # Check required dataset paths
        if self.config['dataset_root'] is None:
            if any([
                self.config['train_images'] is None,
                self.config['train_labels'] is None,
                self.config['val_images'] is None,
                self.config['val_labels'] is None
            ]):
                errors.append(
                    "Either 'dataset_root' or all of "
                    "'train_images', 'train_labels', 'val_images', 'val_labels' must be specified"
                )

        # Check positive values
        if self.config['epochs'] <= 0:
            errors.append("'epochs' must be positive")
        if self.config['batch_size'] <= 0:
            errors.append("'batch_size' must be positive")
        if self.config['learning_rate'] <= 0:
            errors.append("'learning_rate' must be positive")

        # Check valid choices
        valid_optimizers = {'sgd', 'adam', 'adamw', 'rmsprop'}
        if self.config['optimizer'] not in valid_optimizers:
            errors.append(f"'optimizer' must be one of {valid_optimizers}")

        valid_schedulers = {'step', 'cosine',
                            'plateau', 'onecycle', None, 'none'}
        if self.config['scheduler'] not in valid_schedulers:
            errors.append(f"'scheduler' must be one of {valid_schedulers}")

        if errors:
            error_msg = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"Configuration validation failed:\n{error_msg}")

        logger.info("✅ Configuration validation passed")

    def auto_configure_dataset(self, dataset_root: str):
        """
        Auto-configure dataset paths from root directory.

        Assumes standard structure:
        dataset_root/
            train/
                images/
                labels/
            val/
                images/
                labels/
            test/ (optional)
                images/
                labels/

        Args:
            dataset_root: Root directory of dataset
        """
        root = Path(dataset_root)

        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        # Training data
        train_dir = root / 'train'
        if train_dir.exists():
            self.config['train_images'] = str(train_dir / 'images')
            self.config['train_labels'] = str(train_dir / 'labels')

        # Validation data
        val_dir = root / 'val'
        if val_dir.exists():
            self.config['val_images'] = str(val_dir / 'images')
            self.config['val_labels'] = str(val_dir / 'labels')

        # Test data (optional)
        test_dir = root / 'test'
        if test_dir.exists():
            self.config['test_images'] = str(test_dir / 'images')
            self.config['test_labels'] = str(test_dir / 'labels')

        self.config['dataset_root'] = str(root)

        logger.info(f"Auto-configured dataset from {root}")

    def __str__(self) -> str:
        """String representation."""
        lines = ["Training Configuration:"]
        lines.append("=" * 60)

        # Group by category
        categories = {
            'Model': ['model', 'num_classes', 'pretrained', 'freeze_backbone'],
            'Dataset': ['dataset_root', 'train_images', 'val_images', 'image_size'],
            'Training': ['epochs', 'batch_size', 'learning_rate', 'optimizer', 'scheduler'],
            'Loss Weights': ['box_weight', 'cls_weight', 'obj_weight'],
            'Augmentation': ['augment'],
            'Device': ['device', 'num_workers'],
            'Output': ['output_dir', 'experiment_name'],
        }

        for category, keys in categories.items():
            lines.append(f"\n{category}:")
            for key in keys:
                value = self.config.get(key)
                if value is not None:
                    lines.append(f"  {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create comprehensive argument parser with zero hardcoding.

    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Universal Detection Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration file
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON). '
             'If provided, other arguments will override config file values.'
    )

    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model', '-m',
        type=str,
        choices=['yolov8', 'faster_rcnn', 'retinanet', 'fcos'],
        help='Model architecture'
    )
    model_group.add_argument(
        '--num-classes', '-nc',
        type=int,
        help='Number of object classes'
    )
    model_group.add_argument(
        '--pretrained',
        action='store_true',
        help='Use pretrained backbone'
    )
    model_group.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze backbone weights'
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument(
        '--dataset-root', '-dr',
        type=str,
        help='Root directory of dataset (auto-configures train/val/test paths)'
    )
    dataset_group.add_argument(
        '--train-images',
        type=str,
        help='Path to training images directory'
    )
    dataset_group.add_argument(
        '--train-labels',
        type=str,
        help='Path to training labels directory'
    )
    dataset_group.add_argument(
        '--val-images',
        type=str,
        help='Path to validation images directory'
    )
    dataset_group.add_argument(
        '--val-labels',
        type=str,
        help='Path to validation labels directory'
    )
    dataset_group.add_argument(
        '--image-size', '-is',
        type=int,
        help='Input image size (square)'
    )
    dataset_group.add_argument(
        '--label-format', '-lf',
        type=str,
        choices=['yolo', 'coco', 'voc'],
        help='Label file format'
    )

    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of training epochs'
    )
    train_group.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Training batch size'
    )
    train_group.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Initial learning rate'
    )
    train_group.add_argument(
        '--weight-decay', '-wd',
        type=float,
        help='Weight decay (L2 regularization)'
    )
    train_group.add_argument(
        '--momentum',
        type=float,
        help='Momentum for SGD optimizer'
    )

    # Optimizer configuration
    optim_group = parser.add_argument_group('Optimizer Configuration')
    optim_group.add_argument(
        '--optimizer', '-opt',
        type=str,
        choices=['sgd', 'adam', 'adamw', 'rmsprop'],
        help='Optimizer type'
    )
    optim_group.add_argument(
        '--optimizer-params',
        type=str,
        help='Additional optimizer parameters as JSON string. '
             'Example: \'{"betas": [0.9, 0.999], "eps": 1e-8}\''
    )

    # Scheduler configuration
    sched_group = parser.add_argument_group('Scheduler Configuration')
    sched_group.add_argument(
        '--scheduler', '-sch',
        type=str,
        choices=['step', 'cosine', 'plateau', 'onecycle', 'none'],
        help='Learning rate scheduler'
    )
    sched_group.add_argument(
        '--scheduler-params',
        type=str,
        help='Scheduler parameters as JSON string. '
             'Example: \'{"step_size": 30, "gamma": 0.1}\' for StepLR'
    )

    # Loss weights
    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument(
        '--box-weight',
        type=float,
        help='Weight for box regression loss'
    )
    loss_group.add_argument(
        '--cls-weight',
        type=float,
        help='Weight for classification loss'
    )
    loss_group.add_argument(
        '--obj-weight',
        type=float,
        help='Weight for objectness loss'
    )

    # Augmentation
    aug_group = parser.add_argument_group('Data Augmentation')
    aug_group.add_argument(
        '--augment', '-aug',
        action='store_true',
        help='Enable data augmentation'
    )
    aug_group.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation'
    )
    aug_group.add_argument(
        '--augment-params',
        type=str,
        help='Augmentation parameters as JSON string'
    )

    # Regularization
    reg_group = parser.add_argument_group('Regularization')
    reg_group.add_argument(
        '--dropout',
        type=float,
        help='Dropout probability'
    )
    reg_group.add_argument(
        '--label-smoothing',
        type=float,
        help='Label smoothing factor'
    )
    reg_group.add_argument(
        '--mixup-alpha',
        type=float,
        help='Mixup alpha parameter (0 to disable)'
    )
    reg_group.add_argument(
        '--gradient-clip',
        type=float,
        help='Gradient clipping threshold'
    )

    # Training strategy
    strategy_group = parser.add_argument_group('Training Strategy')
    strategy_group.add_argument(
        '--warmup-epochs',
        type=int,
        help='Number of warmup epochs'
    )
    strategy_group.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping'
    )
    strategy_group.add_argument(
        '--early-stopping-patience',
        type=int,
        help='Early stopping patience (epochs)'
    )
    strategy_group.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training'
    )
    strategy_group.add_argument(
        '--no-mixed-precision',
        action='store_true',
        help='Disable mixed precision training'
    )

    # Validation
    val_group = parser.add_argument_group('Validation')
    val_group.add_argument(
        '--val-interval',
        type=int,
        help='Validation interval (epochs)'
    )
    val_group.add_argument(
        '--val-metric',
        type=str,
        choices=['mAP', 'loss', 'f1'],
        help='Metric for best model selection'
    )

    # Checkpointing
    ckpt_group = parser.add_argument_group('Checkpointing')
    ckpt_group.add_argument(
        '--save-interval',
        type=int,
        help='Checkpoint save interval (epochs)'
    )
    ckpt_group.add_argument(
        '--save-best-only',
        action='store_true',
        help='Only save best checkpoint'
    )
    ckpt_group.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory to save checkpoints'
    )
    ckpt_group.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )

    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results'
    )
    output_group.add_argument(
        '--experiment-name', '-n',
        type=str,
        help='Name of experiment (creates subdirectory in output-dir)'
    )
    output_group.add_argument(
        '--log-interval',
        type=int,
        help='Logging interval (iterations)'
    )
    output_group.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save validation predictions'
    )

    # Device configuration
    device_group = parser.add_argument_group('Device Configuration')
    device_group.add_argument(
        '--device', '-d',
        type=str,
        help='Device to use (cuda, cpu, cuda:0, etc.)'
    )
    device_group.add_argument(
        '--num-workers', '-nw',
        type=int,
        help='Number of dataloader workers'
    )
    device_group.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='Disable pin memory for dataloaders'
    )

    # Debugging
    debug_group = parser.add_argument_group('Debugging')
    debug_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    debug_group.add_argument(
        '--profile',
        action='store_true',
        help='Enable profiling'
    )
    debug_group.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    return parser


def parse_args_and_create_config() -> TrainingConfig:
    """
    Parse command-line arguments and create training configuration.

    Returns:
        TrainingConfig instance
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Load from config file if provided
    if args.config:
        config = TrainingConfig.from_file(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = TrainingConfig()

    # Override with command-line arguments
    args_dict = vars(args)

    # Handle special cases
    if args.no_augment:
        args_dict['augment'] = False
    if args.no_mixed_precision:
        args_dict['mixed_precision'] = False
    if args.no_pin_memory:
        args_dict['pin_memory'] = False

    # Parse JSON strings
    if args.optimizer_params:
        args_dict['optimizer_params'] = json.loads(args.optimizer_params)
    if args.scheduler_params:
        args_dict['scheduler_params'] = json.loads(args.scheduler_params)
    if args.augment_params:
        args_dict['augment_params'] = json.loads(args.augment_params)

    # Remove special flag keys that shouldn't be in config
    for key in ['no_augment', 'no_mixed_precision', 'no_pin_memory']:
        args_dict.pop(key, None)

    # Update config with args
    config.update({k: v for k, v in args_dict.items() if v is not None})

    # Auto-configure dataset if root provided
    if config['dataset_root']:
        config.auto_configure_dataset(config['dataset_root'])

    # Validate configuration
    config.validate()

    return config

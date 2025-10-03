"""
Universal Configuration Loader.

Handles the universal config.yaml with short naming conventions
and converts to the format expected by the training system.

Features:
- Short, intuitive config keys (batch, lr, opt, sched, etc.)
- Auto-dataset detection and path construction
- Easy to switch between datasets
- Programmatic and scalable
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and process universal configuration."""

    # Mapping from short names to full names
    SHORT_TO_FULL = {
        # Training
        'batch': 'batch_size',
        'lr': 'learning_rate',
        'wd': 'weight_decay',
        'opt': 'optimizer',
        'opt_params': 'optimizer_params',
        'sched': 'scheduler',
        'sched_params': 'scheduler_params',

        # Augmentation
        'h_flip': 'horizontal_flip',
        'v_flip': 'vertical_flip',
        'rotate': 'rotation',

        # Regularization
        'label_smooth': 'label_smoothing',
        'mixup': 'mixup_alpha',
        'grad_clip': 'gradient_clip',

        # Strategy
        'warmup': 'warmup_epochs',
        'early_stop': 'early_stopping',
        'early_stop_patience': 'early_stopping_patience',

        # Checkpoint
        'ckpt': 'checkpoint',
        'save_best': 'save_best_only',

        # Output
        'exp_name': 'experiment_name',
        'save_pred': 'save_predictions',
        'save_vis': 'save_visualizations',

        # Device
        'workers': 'num_workers',

        # Visualization
        'vis': 'visualization',
        'conf_thresh': 'confidence_threshold',
    }

    def __init__(self, config_path: str = 'config.yaml', dataset_override: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to universal config file
            dataset_override: Override dataset name (e.g., for CLI usage)
        """
        self.config_path = Path(config_path)
        self.dataset_override = dataset_override
        self.raw_config = self._load_yaml()
        self.processed_config = self._process_config()

    def _load_yaml(self) -> Dict:
        """Load YAML config file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _process_config(self) -> Dict[str, Any]:
        """Process config: expand short names, auto-detect paths."""
        config = {}

        # Override dataset if specified
        if self.dataset_override:
            self.raw_config['dataset']['name'] = self.dataset_override
            self.raw_config['dataset']['root'] = f"dataset/{self.dataset_override}"

        # Process dataset configuration
        self._process_dataset(config)

        # Process model configuration
        self._process_model(config)

        # Process training configuration
        self._process_training(config)

        # Process augmentation
        self._process_augmentation(config)

        # Process regularization
        self._process_regularization(config)

        # Process strategy
        self._process_strategy(config)

        # Process validation
        self._process_validation(config)

        # Process checkpointing
        self._process_checkpointing(config)

        # Process output
        self._process_output(config)

        # Process device
        self._process_device(config)

        # Process debug
        self._process_debug(config)

        # Process visualization
        self._process_visualization(config)

        return config

    def _process_dataset(self, config: Dict):
        """Process dataset configuration with auto-path construction."""
        ds_config = self.raw_config['dataset']

        config['dataset_name'] = ds_config['name']
        config['dataset_root'] = ds_config['root']
        config['label_format'] = ds_config.get('format', 'yolo')

        # Auto-construct paths
        root = Path(ds_config['root'])

        # Standard YOLO structure
        config['train_images'] = str(root / 'train' / 'images')
        config['train_labels'] = str(root / 'train' / 'labels')
        config['val_images'] = str(root / 'val' / 'images')
        config['val_labels'] = str(root / 'val' / 'labels')
        config['test_images'] = str(root / 'test' / 'images')
        config['test_labels'] = str(root / 'test' / 'labels')

        # Image size
        config['image_size'] = ds_config.get('image_size', 640)

        # num_classes will be auto-detected by training script
        config['num_classes'] = ds_config.get('num_classes')

    def _process_model(self, config: Dict):
        """Process model configuration."""
        model_config = self.raw_config['model']

        config['model'] = model_config['name']
        config['pretrained'] = model_config.get('pretrained', False)
        config['freeze_backbone'] = model_config.get('freeze_backbone', False)

    def _process_training(self, config: Dict):
        """Process training configuration."""
        train_config = self.raw_config['train']

        # Basic training params (with short name mapping)
        config['epochs'] = train_config['epochs']
        config['batch_size'] = train_config['batch']
        config['learning_rate'] = train_config['lr']
        config['weight_decay'] = train_config.get('wd', 0.0001)
        config['momentum'] = train_config.get('momentum', 0.9)

        # Optimizer
        config['optimizer'] = train_config['opt']
        config['optimizer_params'] = train_config.get('opt_params', {})

        # Scheduler
        config['scheduler'] = train_config.get('sched')
        config['scheduler_params'] = train_config.get('sched_params', {})

        # Loss weights
        loss_config = train_config.get('loss', {})
        config['box_weight'] = loss_config.get('box', 7.5)
        config['cls_weight'] = loss_config.get('cls', 0.5)
        config['obj_weight'] = loss_config.get('obj', 1.0)

    def _process_augmentation(self, config: Dict):
        """Process augmentation configuration."""
        aug_config = self.raw_config.get('aug', {})

        config['augment'] = aug_config.get('enabled', True)
        config['augment_params'] = {
            'horizontal_flip': aug_config.get('h_flip', 0.5),
            'vertical_flip': aug_config.get('v_flip', 0.0),
            'rotation': aug_config.get('rotate', 10),
            'brightness': aug_config.get('brightness', 0.2),
            'contrast': aug_config.get('contrast', 0.2),
            'saturation': aug_config.get('saturation', 0.2),
            'hue': aug_config.get('hue', 0.1),
        }

    def _process_regularization(self, config: Dict):
        """Process regularization configuration."""
        reg_config = self.raw_config.get('reg', {})

        config['dropout'] = reg_config.get('dropout', 0.3)
        config['label_smoothing'] = reg_config.get('label_smooth', 0.0)
        config['mixup_alpha'] = reg_config.get('mixup', 0.0)
        config['gradient_clip'] = reg_config.get('grad_clip', 1.0)

    def _process_strategy(self, config: Dict):
        """Process training strategy configuration."""
        strategy_config = self.raw_config.get('strategy', {})

        config['warmup_epochs'] = strategy_config.get('warmup', 5)
        config['early_stopping'] = strategy_config.get('early_stop', True)
        config['early_stopping_patience'] = strategy_config.get(
            'early_stop_patience', 20)
        config['mixed_precision'] = strategy_config.get(
            'mixed_precision', True)

    def _process_validation(self, config: Dict):
        """Process validation configuration."""
        val_config = self.raw_config.get('val', {})

        config['val_interval'] = val_config.get('interval', 1)
        config['val_metric'] = val_config.get('metric', 'mAP')

    def _process_checkpointing(self, config: Dict):
        """Process checkpointing configuration."""
        ckpt_config = self.raw_config.get('ckpt', {})

        config['save_interval'] = ckpt_config.get('save_interval', 5)
        config['save_best_only'] = ckpt_config.get('save_best', False)
        config['checkpoint_dir'] = ckpt_config.get('dir')

    def _process_output(self, config: Dict):
        """Process output configuration."""
        output_config = self.raw_config.get('output', {})

        config['output_dir'] = output_config.get('dir', 'outputs')
        config['experiment_name'] = output_config.get('exp_name')
        config['log_interval'] = output_config.get('log_interval', 10)
        config['save_predictions'] = output_config.get('save_pred', True)
        config['save_visualizations'] = output_config.get('save_vis', True)

    def _process_device(self, config: Dict):
        """Process device configuration."""
        device_config = self.raw_config.get('device', {})

        config['device'] = device_config.get('name', 'cuda')
        config['num_workers'] = device_config.get('workers', 4)
        config['pin_memory'] = device_config.get('pin_memory', True)

    def _process_debug(self, config: Dict):
        """Process debug configuration."""
        debug_config = self.raw_config.get('debug', {})

        config['debug'] = debug_config.get('enabled', False)
        config['profile'] = debug_config.get('profile', False)
        config['seed'] = debug_config.get('seed', 42)

    def _process_visualization(self, config: Dict):
        """Process visualization configuration."""
        vis_config = self.raw_config.get('vis', {})

        config['vis_enabled'] = vis_config.get('enabled', True)
        config['vis_interval'] = vis_config.get('save_interval', 10)
        config['vis_num_samples'] = vis_config.get('num_samples', 8)
        config['vis_show_gt'] = vis_config.get('show_gt', True)
        config['vis_show_pred'] = vis_config.get('show_pred', True)
        config['vis_conf_thresh'] = vis_config.get('conf_thresh', 0.5)

    def get_config(self) -> Dict[str, Any]:
        """Get processed configuration."""
        return self.processed_config

    def save_processed_config(self, output_path: str):
        """Save processed config for reference."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self.processed_config, f,
                      default_flow_style=False, sort_keys=False)

        logger.info(f"Processed config saved to: {output_path}")

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigLoader(dataset={self.processed_config.get('dataset_name')}, model={self.processed_config.get('model')})"


def load_universal_config(config_path: str = 'config.yaml',
                          dataset: Optional[str] = None) -> Dict[str, Any]:
    """
    Load universal configuration.

    Args:
        config_path: Path to config.yaml
        dataset: Override dataset name

    Returns:
        Processed configuration dictionary

    Example:
        # Load default config
        config = load_universal_config()

        # Load with dataset override
        config = load_universal_config(dataset='cattleface')
    """
    loader = ConfigLoader(config_path, dataset)
    return loader.get_config()


if __name__ == '__main__':
    # Test the config loader
    print("Testing ConfigLoader...")

    loader = ConfigLoader('config.yaml')
    config = loader.get_config()

    print(f"\n{loader}")
    print(f"\nDataset: {config['dataset_name']}")
    print(f"Model: {config['model']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Scheduler: {config['scheduler']}")

    print("\n✅ Config loader working correctly!")

"""
Configuration manager that merges YAML + CLI args.
Priority: CLI args > config.yaml > defaults
"""

import yaml
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, Optional

from .defaults import DEFAULTS


class ConfigManager:
    """
    Manages configuration from multiple sources with priority:
    CLI args > config.yaml > defaults
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.

        Args:
            config_path: Path to YAML config file. If None, uses default path.
        """
        if config_path is None:
            config_path = 'src/config/config.yaml'

        self.config_path = Path(config_path)
        self.yaml_config = self._load_yaml()
        self.defaults = deepcopy(DEFAULTS)

    def _load_yaml(self) -> Dict[str, Any]:
        """
        Load YAML config file.

        Returns:
            Dict containing YAML config, empty dict if file doesn't exist
        """
        if not self.config_path.exists():
            print(
                f"Warning: Config file not found at {self.config_path}, using defaults")
            return {}

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                return config if config is not None else {}
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def merge_config(self, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configs with priority: CLI args > YAML > Defaults

        Args:
            cli_args: Dictionary of CLI arguments

        Returns:
            Merged configuration dictionary
        """
        # Start with defaults
        config = deepcopy(self.defaults)

        # Override with YAML
        self._deep_update(config, self.yaml_config)

        # Override with CLI args (remove None values)
        cli_clean = self._clean_cli_args(cli_args)
        self._apply_cli_overrides(config, cli_clean)

        # Add metadata
        config['_metadata'] = {
            'config_file': str(self.config_path),
            'cli_args': cli_clean,
        }

        return config

    def _clean_cli_args(self, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove None values and command from CLI args.

        Args:
            cli_args: Raw CLI arguments

        Returns:
            Cleaned CLI arguments
        """
        return {k: v for k, v in cli_args.items()
                if v is not None and k != 'command'}

    def _apply_cli_overrides(self, config: Dict[str, Any], cli_args: Dict[str, Any]):
        """
        Apply CLI argument overrides to config.

        Args:
            config: Base configuration to update
            cli_args: CLI arguments to apply
        """
        # Direct mappings
        direct_mappings = {
            'model': 'model',
            'dataset': 'dataset',
            'epochs': 'training.epochs',
            'batch_size': 'training.batch_size',
            'learning_rate': 'training.learning_rate',
            'weight_decay': 'training.weight_decay',
            'optimizer': 'training.optimizer',
            'num_workers': 'training.num_workers',
            'checkpoint_freq': 'training.checkpoint_freq',
            'log_freq': 'training.log_freq',
            'mixed_precision': 'training.mixed_precision',
            'seed': 'training.seed',
            'img_size': 'data.img_size',
            'device': 'device.type',
            'iou_threshold': 'evaluation.iou_threshold',
            'conf_threshold': 'evaluation.conf_threshold',
            'save_predictions': 'evaluation.save_predictions',
        }

        for cli_key, config_path in direct_mappings.items():
            if cli_key in cli_args:
                self._set_nested_value(config, config_path, cli_args[cli_key])

        # Special handling for resume checkpoint
        if 'resume' in cli_args and cli_args['resume']:
            config['resume_checkpoint'] = cli_args['resume']

        # Special handling for custom config file
        if 'config' in cli_args and cli_args['config']:
            # Already handled in __init__
            pass

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """
        Set a nested dictionary value using dot notation.

        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., 'training.epochs')
            value: Value to set
        """
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any]):
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary to update
            update: Dictionary with updates
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def save_config(self, config: Dict[str, Any], output_path: Path):
        """
        Save merged config to file for reproducibility.

        Args:
            config: Configuration to save
            output_path: Path to save config file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove metadata before saving
        config_to_save = deepcopy(config)
        config_to_save.pop('_metadata', None)

        with open(output_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, indent=2)

        print(f"Config saved to {output_path}")

    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        """
        Load a saved config file.

        Args:
            config_path: Path to config file

        Returns:
            Loaded configuration
        """
        with open(config_path) as f:
            return yaml.safe_load(f)


# Convenience function
def get_config(cli_args: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get merged configuration.

    Args:
        cli_args: CLI arguments dictionary
        config_path: Optional custom config file path

    Returns:
        Merged configuration
    """
    # Use custom config if provided in CLI args
    if 'config' in cli_args and cli_args['config']:
        config_path = cli_args['config']

    manager = ConfigManager(config_path)
    return manager.merge_config(cli_args)

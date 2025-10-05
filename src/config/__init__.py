# Import from new modular config system
from .manager import ConfigManager, get_config
from .defaults import DEFAULTS

__all__ = [
    'ConfigManager',
    'get_config',
    'DEFAULTS',
]

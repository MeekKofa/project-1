# Import from legacy config files for backward compatibility
from .paths import *
from .hyperparameters import *

# Import from main settings file (now in same directory)
from .settings import (
    TRAINING_CONFIGS,
    EVALUATION_CONFIGS,
    DEFAULT_HYPERPARAMETERS,
    OUTPUT_LOGS_DIR,
    OUTPUT_MODELS_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_RESULTS_DIR,
    get_config,
    ensure_output_dirs,
    create_output_dir,
    get_output_path,
    get_systematic_output_dir,
    get_model_config
)

# Make main config items available at package level
__all__ = [
    'TRAINING_CONFIGS',
    'EVALUATION_CONFIGS',
    'DEFAULT_HYPERPARAMETERS',
    'OUTPUT_LOGS_DIR',
    'OUTPUT_MODELS_DIR',
    'OUTPUT_IMAGES_DIR',
    'OUTPUT_RESULTS_DIR',
    'get_config',
    'ensure_output_dirs',
    'create_output_dir',
    'get_output_path',
    'get_systematic_output_dir',
    'get_model_config'
]

"""
Universal Training Script - Zero Hardcoding.

Complete training pipeline with comprehensive argument parsing.
Works with any model, dataset, optimizer, scheduler, etc.

Example usage:
    # Basic training
    python train_universal.py \\
        --model yolov8 \\
        --dataset-root dataset/cattlebody \\
        --num-classes 2 \\
        --epochs 100 \\
        --batch-size 8
    
    # Advanced training with custom optimizer and scheduler
    python train_universal.py \\
        --model yolov8 \\
        --dataset-root dataset/cattlebody \\
        --num-classes 2 \\
        --optimizer adamw \\
        --learning-rate 1e-3 \\
        --scheduler cosine \\
        --scheduler-params '{"T_max": 100, "eta_min": 1e-6}' \\
        --augment \\
        --mixed-precision \\
        --early-stopping
    
    # Training with config file
    python train_universal.py --config configs/yolov8_cattlebody.yaml
    
    # Override config file with command-line args
    python train_universal.py \\
        --config configs/base.yaml \\
        --learning-rate 2e-3 \\
        --epochs 200
"""

# CRITICAL: Setup Python path FIRST, before any src imports
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import everything else
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import json
import yaml

from src.config.training_config import TrainingConfig, parse_args_and_create_config
from src.core.registry import ModelRegistry
from src.data.detection_dataset import DetectionDataset
from src.training.trainer import DetectionTrainer
from src.utils.logging_utils import setup_logging

# Import models to register them
from src.models.yolov8.architecture import YOLOv8Model

# Add project root to Python path (must be before src imports)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Add project root to Python path (must be before src imports)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Import models to register them (must be after ModelRegistry import)

logger = logging.getLogger(__name__)


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        Optimizer instance
    """
    optimizer_type = config['optimizer'].lower()
    lr = config['learning_rate']
    weight_decay = config['weight_decay']

    # Get additional optimizer parameters
    optimizer_params = config.get('optimizer_params', {})

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
            **optimizer_params
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_params
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_params
        )
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_params
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    logger.info(f"Created optimizer: {optimizer_type}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Weight decay: {weight_decay}")
    if optimizer_params:
        logger.info(f"  Additional params: {optimizer_params}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: Optimizer instance
        config: Training configuration

    Returns:
        Scheduler instance or None
    """
    scheduler_type = config.get('scheduler')

    if scheduler_type is None or scheduler_type == 'none':
        logger.info("No learning rate scheduler")
        return None

    scheduler_params = config.get('scheduler_params', {})

    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 30),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('T_max', config['epochs']),
            eta_min=scheduler_params.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=scheduler_params.get('factor', 0.1),
            patience=scheduler_params.get('patience', 10),
            verbose=True
        )
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            total_steps=scheduler_params.get('total_steps', config['epochs']),
            pct_start=scheduler_params.get('pct_start', 0.3)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    logger.info(f"Created scheduler: {scheduler_type}")
    if scheduler_params:
        logger.info(f"  Scheduler params: {scheduler_params}")

    return scheduler


def create_dataloaders(config: TrainingConfig) -> tuple:
    """
    Create training and validation dataloaders.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating dataloaders...")

    # Training dataset
    train_dataset = DetectionDataset(
        images_dir=config['train_images'],
        labels_dir=config['train_labels'],
        image_size=config['image_size'],
        augment=config['augment'],
        normalize=True
    )

    # Validation dataset
    val_dataset = DetectionDataset(
        images_dir=config['val_images'],
        labels_dir=config['val_labels'],
        image_size=config['image_size'],
        augment=False,
        normalize=True
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=DetectionDataset.collate_fn,
        pin_memory=config.get('pin_memory', True),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=DetectionDataset.collate_fn,
        pin_memory=config.get('pin_memory', True)
    )

    return train_loader, val_loader


def setup_output_dir(config: TrainingConfig) -> Path:
    """
    Setup output directory structure.

    Args:
        config: Training configuration

    Returns:
        Path to output directory
    """
    output_dir = Path(config['output_dir'])

    # Add experiment name if provided
    if config.get('experiment_name'):
        output_dir = output_dir / config['experiment_name']
    else:
        # Auto-generate experiment name
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{config['model']}_{timestamp}"
        output_dir = output_dir / experiment_name

    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'metrics',
               'predictions', 'visualizations']
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Save configuration
    config_path = output_dir / 'config.yaml'
    config.save(str(config_path))
    logger.info(f"Configuration saved to {config_path}")

    return output_dir


def main():
    """Main training function."""
    # Parse arguments and create configuration
    config = parse_args_and_create_config()

    # Setup output directory
    output_dir = setup_output_dir(config)

    # Setup logging
    log_file = output_dir / 'logs' / 'training.log'
    setup_logging(log_file=str(log_file),
                  level=logging.INFO if not config['debug'] else logging.DEBUG)

    logger.info("=" * 80)
    logger.info("UNIVERSAL DETECTION TRAINING")
    logger.info("=" * 80)
    logger.info(str(config))
    logger.info("=" * 80)

    # Set random seed for reproducibility
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
        logger.info(f"Random seed set to {config['seed']}")

    # Setup device
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    # Create dataloaders FIRST to auto-detect num_classes
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Auto-detect num_classes from dataset if not provided
    if config['num_classes'] is None:
        # Get num_classes from the dataset
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'num_classes'):
            config['num_classes'] = train_dataset.num_classes
        else:
            # Try to detect from data.yaml or labels
            data_yaml_path = Path(config['dataset_root']) / 'data.yaml'
            if data_yaml_path.exists():
                with open(data_yaml_path, 'r') as f:
                    data_info = yaml.safe_load(f)
                    config['num_classes'] = data_info.get('nc', len(data_info.get('names', [])))
            else:
                # Last resort: scan label files
                logger.warning("Could not auto-detect num_classes, scanning label files...")
                max_class = 0
                label_dir = Path(config['train_labels'])
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                max_class = max(max_class, class_id)
                config['num_classes'] = max_class + 1
        
        logger.info(f"Auto-detected num_classes: {config['num_classes']}")
    else:
        logger.info(f"Using provided num_classes: {config['num_classes']}")

    # Create model
    logger.info(f"Creating model: {config['model']}")

    # Build model configuration
    # YOLOv8 accepts: num_classes, in_channels, base_channels, config
    model_config = {
        'num_classes': config['num_classes'],
    }

    # Add optional parameters if provided
    if 'in_channels' in config:
        model_config['in_channels'] = config['in_channels']
    if 'base_channels' in config:
        model_config['base_channels'] = config['base_channels']

    # Pass model-specific config (loss weights, etc.) as nested config dict
    if config['model'] == 'yolov8':
        model_config['config'] = {
            'box_weight': config['box_weight'],
            'cls_weight': config['cls_weight'],
            'obj_weight': config['obj_weight'],
            'dropout': config.get('dropout', 0.3),
        }

    model = ModelRegistry.build(config['model'], **model_config)
    model = model.to(device)

    # Print model info
    model_info = model.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    # Freeze backbone if requested
    if config.get('freeze_backbone', False):
        model.freeze_backbone(True)
        logger.info("Backbone frozen")

    # Dataloaders already created above for num_classes detection

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Create scheduler
    scheduler = create_scheduler(optimizer, config)

    # Create trainer
    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        output_dir=str(output_dir),
        lr_scheduler=scheduler,
        grad_clip=config.get('gradient_clip'),
        log_interval=config['log_interval'],
        val_interval=config['val_interval']
    )

    # Training configuration for trainer
    trainer_config = {
        'epochs': config['epochs'],
        'early_stopping': config.get('early_stopping', False),
        'early_stopping_patience': config.get('early_stopping_patience', 20),
        'save_interval': config.get('save_interval', 5),
        'save_best_only': config.get('save_best_only', False),
        'val_metric': config.get('val_metric', 'mAP'),
        'mixed_precision': config.get('mixed_precision', True),
    }

    logger.info("Starting training...")
    logger.info("=" * 80)

    # Train model
    history = trainer.train(num_epochs=config['epochs'])

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)

    # Print final metrics
    logger.info("Final Training Metrics:")
    for key, value in history['train_metrics'].items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("\nFinal Validation Metrics:")
    for key, value in history['val_metrics'].items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info(f"\nBest {config.get('val_metric', 'mAP')}: {history['best_metric']:.4f} "
                f"(epoch {history['best_epoch']})")

    # Save final history
    history_path = output_dir / 'metrics' / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"\nTraining history saved to {history_path}")

    logger.info("=" * 80)
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)

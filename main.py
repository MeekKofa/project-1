#!/usr/bin/env python3
"""
Main entry point for the Cattle Detection and Recognition System.

This script provides a com        # Prepare training arguments
        train_args = {
            'dataset': dataset,
            'output_dir': str(models_dir),
            'log_file': str(log_file),
            'metrics_dir': str(metrics_dir),
            'results_dir': str(results_dir),
            'images_dir': str(images_dir),
            'checkpoints_dir': str(checkpoints_dir)
        }ine interface to run different training models
and evaluation tasks. All outputs are organized in the outputs/ directory.

Usage Examples:
    # Train Faster R-CNN on cattlebody dataset
    python main.py train --model faster_rcnn --dataset cattlebody
    
    # Train YOLOv8 with custom epochs and batch size
    python main.py train --model yolov8 --dataset cattleface --epochs 50 --batch-size 16
    
    # Train on specific GPU
    python main.py train --model faster_rcnn --dataset cattle --device 1
    python main.py train --model yolov8 --dataset cattleface --device cuda:2


    python main.py train -m yolov8 -d cattle -e 2 -b 8 --device cuda:1
    
    # Run model evaluation
    python main.py evaluate --model faster_rcnn --dataset cattleface
    
    # Show available devices
    python main.py info --list-devices
    
    # Show project structure
    python main.py info --show-structure
    
    # List available models and datasets
    python main.py info --list-models
"""

from src.config import (
    TRAINING_CONFIGS,
    EVALUATION_CONFIGS,
    OUTPUT_LOGS_DIR,
    get_config,
    create_output_dir,
    get_output_path,
    get_systematic_output_dir
)
import sys
import os
from pathlib import Path
import argparse
import logging
import importlib
import torch
from typing import Optional, Dict, Any

# Add src to path for imports FIRST - before any local imports
current_dir = Path(__file__).parent.absolute()
src_path = str(current_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now we can import from config package


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create logs directory only when setting up logging (this is necessary)
    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_LOGS_DIR / "main.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_available_datasets() -> Dict[str, str]:
    """Get available datasets."""
    return {
        "cattlebody": "Cattle body detection dataset",
        "cattleface": "Cattle face detection dataset",
        "cattle": "Combination of cattleface and catle body"
    }


def get_available_models() -> Dict[str, str]:
    """Get available models from training configs."""
    return {key: config["description"] for key, config in TRAINING_CONFIGS.items()}


def train_model(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Train a model with the specified configuration."""
    model_name = args.model

    # Handle new dataset specification methods
    if hasattr(args, 'dataset_path') and args.dataset_path:
        # Robust mode: using direct dataset path
        dataset_path = args.dataset_path
        # Use folder name as dataset identifier
        dataset_name = Path(dataset_path).name
        logger.info(f"Using dataset path: {dataset_path}")

        # Validate dataset exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return False

    else:
        # Backward compatibility mode: using dataset name
        dataset_name = args.dataset
        if dataset_name not in get_available_datasets():
            logger.error(
                f"Unknown dataset: {dataset_name}. Available datasets: {list(get_available_datasets().keys())}")
            return False
        dataset_path = None  # Will be determined by training function
        logger.info(
            f"Using dataset name: {dataset_name} (backward compatibility)")

    if model_name not in TRAINING_CONFIGS:
        logger.error(
            f"Unknown model: {model_name}. Available models: {list(TRAINING_CONFIGS.keys())}")
        return False

    config = TRAINING_CONFIGS[model_name]
    logger.info(
        f"Starting {config['name']} training on {dataset_name} dataset...")
    logger.info(f"Description: {config['description']}")

    # Create systematic output directories: outputs/{dataset}/{model}/{type}/
    models_dir = get_systematic_output_dir(dataset_name, model_name, "models")
    logs_dir = get_systematic_output_dir(dataset_name, model_name, "logs")
    results_dir = get_systematic_output_dir(
        dataset_name, model_name, "results")
    images_dir = get_systematic_output_dir(dataset_name, model_name, "images")
    metrics_dir = get_systematic_output_dir(
        dataset_name, model_name, "metrics")
    checkpoints_dir = get_systematic_output_dir(
        dataset_name, model_name, "checkpoints")

    logger.info(f"Output directory: {models_dir}")
    logger.info(f"Logs directory: {logs_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Metrics directory: {metrics_dir}")

    # Create log file path
    log_file = logs_dir / f"{model_name}_{dataset_name}.log"

    try:
        # Import and run the training module
        module = importlib.import_module(config["module"])

        # Prepare training arguments - support both robust and backward compatibility modes
        train_args = {
            'dataset': dataset_name,
            'output_dir': str(models_dir),
            'log_file': str(log_file),
            'metrics_dir': str(metrics_dir),
            'results_dir': str(results_dir),
            'images_dir': str(images_dir),
            'checkpoints_dir': str(checkpoints_dir)
        }

        # Add robust dataset configuration if using new path-based method
        if hasattr(args, 'dataset_path') and args.dataset_path:
            train_args['dataset_path'] = args.dataset_path
            train_args['validate_dataset'] = getattr(
                args, 'validate_dataset', False)
            if hasattr(args, 'num_classes') and args.num_classes:
                train_args['num_classes'] = args.num_classes

        # Add optional arguments if provided
        if hasattr(args, 'epochs') and args.epochs:
            train_args['epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            train_args['batch_size'] = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            train_args['learning_rate'] = args.learning_rate
        if hasattr(args, 'device') and args.device:
            # Validate and parse the device argument
            try:
                from src.utils.device_utils import parse_device
                device = parse_device(args.device)
                # Convert to string for compatibility
                train_args['device'] = str(device)
                logger.info(f"Using device: {device}")
            except ImportError:
                logger.warning(
                    "Device utils not available, using device string directly")
                train_args['device'] = args.device
            except ValueError as e:
                logger.error(f"Invalid device specification: {e}")
                return False

        # Call the main function of the training module
        if hasattr(module, 'main'):
            result = module.main(**train_args)
            if result:
                logger.info(
                    f"✅ {config['name']} training completed successfully!")
                return True
            else:
                logger.error(f"❌ {config['name']} training failed.")
                return False
        else:
            logger.error(
                f"Training module {config['module']} does not have a main function")
            return False

    except ImportError as e:
        logger.error(f"Failed to import {config['module']}: {e}")
        return False
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return False


def evaluate_model(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Evaluate a trained model using comprehensive metrics."""
    model_name = args.model
    dataset = args.dataset

    if model_name not in TRAINING_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        return False

    config = TRAINING_CONFIGS[model_name]
    logger.info(f"Starting {model_name} evaluation on {dataset} dataset...")
    logger.info(f"Description: {config['description']}")

    try:
        # Import our comprehensive evaluation module
        from src.evaluation.evaluate_model import main as eval_main

        # Determine model path
        if hasattr(args, 'model_path') and args.model_path:
            model_path = args.model_path
        else:
            output_dir = config.get(
                'output_dir', f'./outputs/models/{model_name}')
            model_path = f"{output_dir}.pth"  # Assuming standard naming

            if not os.path.exists(model_path):
                model_path = f"./outputs/models/{model_name}.pth"

        if not os.path.exists(model_path):
            logger.error(f"Model checkpoint not found at {model_path}")
            logger.info(
                "Please train the model first or specify the correct model path")
            return False

        logger.info(f"Using model: {model_path}")

        # Prepare arguments for evaluation script
        if hasattr(args, 'output_dir') and args.output_dir:
            output_eval_dir = args.output_dir
        else:
            output_eval_dir = f'./outputs/evaluation/{model_name}_{dataset}'

        eval_args = [
            '--model', model_path,
            '--dataset', dataset,
            '--output-dir', output_eval_dir,
        ]

        # Add optional arguments if provided
        if hasattr(args, 'batch_size') and args.batch_size:
            eval_args.extend(['--batch-size', str(args.batch_size)])
        if hasattr(args, 'score_threshold') and args.score_threshold:
            eval_args.extend(['--score-threshold', str(args.score_threshold)])

        # Mock sys.argv for the evaluation script
        import sys
        original_argv = sys.argv
        sys.argv = ['evaluate_model.py'] + eval_args

        try:
            result = eval_main()
            if result == 0:
                logger.info("✅ Model evaluation completed successfully!")
                return True
            else:
                logger.error("❌ Model evaluation failed.")
                return False
        finally:
            sys.argv = original_argv

    except ImportError as e:
        logger.error(f"Failed to import evaluation module: {e}")
        return False
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        return False


def preprocess_dataset(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Preprocess raw dataset for training."""
    dataset = args.dataset
    split_ratio = args.split_ratio
    force = args.force

    logger.info(f"🔄 Starting preprocessing of {dataset} dataset...")
    logger.info(f"Train/validation split ratio: {split_ratio:.2f}")

    # Check if output already exists
    output_dir = Path(f"processed_data/{dataset}")
    if output_dir.exists() and not force:
        logger.warning(f"⚠️  Processed dataset already exists at {output_dir}")
        logger.warning(
            "Use --force to reprocess, or remove the directory manually")
        return False

    # Check if raw dataset exists
    raw_dataset_dir = Path(f"dataset/{dataset}")
    if not raw_dataset_dir.exists():
        logger.error(f"❌ Raw dataset not found at {raw_dataset_dir}")
        return False

    try:
        # Import the preprocessing module
        from src.processing.preprocessing import convert_dataset_to_training_format

        # Run preprocessing using the new function
        success = convert_dataset_to_training_format(
            dataset_name=dataset,
            project_root=os.getcwd(),
            force=force
        )

        if success:
            logger.info(f"✅ {dataset} dataset preprocessing completed!")
            logger.info(f"📁 Processed data saved to: {output_dir}")
            return True
        else:
            logger.error("❌ Dataset preprocessing failed.")
            return False

    except ImportError as e:
        logger.error(f"Failed to import preprocessing module: {e}")
        logger.error(
            "💡 Hint: Make sure src/processing/preprocessing.py exists and has convert_dataset_to_training_format function")
        return False
    except Exception as e:
        logger.error(f"Preprocessing failed with error: {e}")
        return False


def visualize_results(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Visualize model results."""
    try:
        from scripts.inference import main as viz_main
        viz_main()
        logger.info("✅ Visualization completed!")
        return True
    except ImportError as e:
        logger.error(f"Failed to import visualization module: {e}")
        return False
    except Exception as e:
        logger.error(f"Visualization failed with error: {e}")
        return False


def run_debug(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Run debug tests with robust dataset configuration."""
    try:
        # Handle new dataset specification methods
        if hasattr(args, 'dataset_path') and args.dataset_path:
            # Robust mode: using direct dataset path
            dataset_path = args.dataset_path
            # Use folder name as dataset identifier
            dataset_name = Path(dataset_path).name
            logger.info(f"Debug using dataset path: {dataset_path}")

            # Validate dataset exists
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset path does not exist: {dataset_path}")
                return False

        else:
            # Backward compatibility mode: using dataset name
            dataset_name = args.dataset
            if dataset_name not in get_available_datasets():
                logger.error(
                    f"Unknown dataset: {dataset_name}. Available datasets: {list(get_available_datasets().keys())}")
                return False
            dataset_path = None  # Will be determined by debug function
            logger.info(
                f"Debug using dataset name: {dataset_name} (backward compatibility)")

        # Import debug module and prepare arguments
        from run_debug_sample import main as debug_main

        # Create debug configuration arguments
        debug_config = {
            'dataset_name': dataset_name,
            'validate_dataset': getattr(args, 'validate_dataset', False),
            'sample_size': getattr(args, 'sample_size', 10),
        }

        # Add robust dataset configuration if using new path-based method
        if hasattr(args, 'dataset_path') and args.dataset_path:
            debug_config['dataset_path'] = args.dataset_path
            if hasattr(args, 'num_classes') and args.num_classes:
                debug_config['num_classes'] = args.num_classes

        # Run debug with configuration
        logger.info("🔍 Starting debug tests...")
        debug_main(**debug_config)
        logger.info("✅ Debug tests completed!")
        return True

    except ImportError as e:
        logger.error(f"Failed to import debug module: {e}")
        return False
    except Exception as e:
        logger.error(f"Debug tests failed with error: {e}")
        logger.exception("Full error details:")
        return False


def show_info(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Show system information."""
    if args.list_models:
        print("\n📋 Available Models:")
        print("=" * 50)
        for key, description in get_available_models().items():
            print(f"  {key:15} - {description}")

        print(f"\n📊 Available Datasets:")
        print("=" * 50)
        for key, description in get_available_datasets().items():
            print(f"  {key:15} - {description}")

    if args.list_devices:
        print("\n🖥️  Available Devices:")
        print("=" * 50)
        try:
            from src.utils.device_utils import list_available_devices, get_device_info, parse_device
            devices = list_available_devices()
            for device_str in devices:
                try:
                    device = parse_device(device_str)
                    info = get_device_info(device)
                    if device.type == 'cuda':
                        print(f"  {device_str:10} - {info.get('gpu_name', 'Unknown GPU')} "
                              f"({info.get('total_memory_gb', 0):.1f}GB)")
                    else:
                        print(f"  {device_str:10} - {device}")
                except Exception:
                    print(f"  {device_str:10} - Available")
        except ImportError:
            logger.warning("Device utils not available")
            print("  cpu        - CPU processing")
            if torch.cuda.is_available():
                print("  cuda       - Default CUDA device")
                print("  auto       - Auto-select best device")

    if args.show_structure:
        print("\n📁 Project Structure:")
        print("=" * 50)
        try:
            import subprocess
            result = subprocess.run(['tree', str(Path.cwd()), '-d', '-L', '3'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("Tree command failed, showing basic structure:")
                print_basic_structure()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print_basic_structure()


def print_basic_structure():
    """Print basic project structure without tree command."""
    structure = """
project/
├── main.py              # Main entry point
├── src/                 # Source code
│   ├── config/         # Configuration files
│   ├── models/         # Model architectures
│   ├── training/       # Training scripts
│   ├── evaluation/     # Evaluation scripts
│   ├── processing/     # Data processing
│   └── utils/          # Utility functions
├── outputs/            # Generated outputs
│   ├── models/        # Trained models
│   ├── logs/          # Log files
│   ├── results/       # Results and metrics
│   └── images/        # Generated images
├── dataset/           # Raw datasets
└── processed_data/    # Processed datasets
    """
    print(structure)


def cleanup_metrics(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """
    Clean up old individual epoch metric files.

    Args:
        args: Command line arguments
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        from src.evaluation.metrics import DetectionMetrics
        from pathlib import Path

        if args.dir:
            # Clean specific directory
            metrics_dir = Path(args.dir)
            if not metrics_dir.exists():
                logger.error(f"Directory not found: {metrics_dir}")
                return False

            logger.info(f"🧹 Cleaning metrics directory: {metrics_dir}")
            DetectionMetrics.cleanup_old_metrics(str(metrics_dir))

        elif args.all:
            # Clean all metrics directories
            outputs_dir = Path('outputs')
            if not outputs_dir.exists():
                logger.error("Outputs directory not found")
                return False

            # Confirmation for non-dry-run operations
            if not args.dry_run and not args.force:
                response = input(
                    "\n⚠️  This will remove individual epoch files from ALL metrics directories. Continue? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    logger.info("Operation cancelled by user")
                    return True

            metrics_dirs = []

            # Find all metrics directories
            for dataset_dir in outputs_dir.iterdir():
                if dataset_dir.is_dir() and dataset_dir.name != 'legacy':
                    for model_dir in dataset_dir.iterdir():
                        if model_dir.is_dir():
                            metrics_dir = model_dir / 'metrics'
                            if metrics_dir.exists():
                                metrics_dirs.append(metrics_dir)

            if not metrics_dirs:
                logger.info("No metrics directories found to clean")
                return True

            logger.info(
                f"🧹 Found {len(metrics_dirs)} metrics directories to clean")

            total_removed = 0
            for metrics_dir in metrics_dirs:
                logger.info(
                    f"Cleaning: {metrics_dir.relative_to(outputs_dir)}")

                if args.dry_run:
                    # Count files that would be removed
                    count = 0
                    for file_path in metrics_dir.iterdir():
                        if file_path.is_file():
                            filename = file_path.name
                            if 'epoch_' in filename and (filename.endswith('.json') or filename.endswith('.txt')):
                                if not any(pattern in filename for pattern in ['.csv', 'final', 'training_curves']):
                                    count += 1
                    logger.info(
                        f"Would remove {count} files from {metrics_dir.relative_to(outputs_dir)}")
                    total_removed += count
                else:
                    DetectionMetrics.cleanup_old_metrics(str(metrics_dir))

            if args.dry_run:
                logger.info(
                    f"🔍 Dry run complete: Would remove {total_removed} files total")
                logger.info("💡 Run without --dry-run to actually remove files")
            else:
                logger.info("🎉 Metrics cleanup completed for all directories!")
                logger.info(
                    "📊 Consolidated metrics are available in training_metrics.csv files")

        else:
            logger.error("Must specify either --dir or --all")
            return False

        return True

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False


def run_hyperparameter_optimization(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Run hyperparameter optimization."""
    try:
        from src.config.hyperparameters import get_hyperparameters

        logger.info(
            f"Starting hyperparameter optimization for {args.model} on {args.dataset}")
        logger.info(
            f"Profile: {args.profile}, Trials: {args.trials}, Max epochs per trial: {args.max_epochs}")

        # Get the specified hyperparameter profile
        hyperparams = get_hyperparameters(args.model, args.profile)
        logger.info(f"Base hyperparameters: {hyperparams}")

        # For now, we'll use the profile-based training
        # In the future, this could integrate with optuna or similar

        # Import the appropriate training function
        if args.model == 'faster_rcnn':
            from src.training.train_faster_rcnn import train_faster_rcnn as train_func
        elif args.model == 'yolov8':
            from src.training.train_yolov8 import train_yolov8 as train_func
        else:
            logger.error(
                f"Optimization not implemented for model: {args.model}")
            return False

        # Run training with optimized hyperparameters
        success = train_func(
            dataset_name=args.dataset,
            **hyperparams,
            num_epochs=args.max_epochs
        )

        if success:
            logger.info(
                "✅ Hyperparameter optimization completed successfully!")
            logger.info("Check outputs directory for results and metrics")
        else:
            logger.error("❌ Hyperparameter optimization failed")

        return success

    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {e}")
        return False


def run_advanced_training(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Run advanced training with optimizations."""
    try:
        from src.config.hyperparameters import get_hyperparameters, DATA_AUG_PARAMS, OPTIMIZATION_PARAMS

        logger.info(
            f"Starting advanced training for {args.model} on {args.dataset}")
        logger.info(f"Profile: {args.profile}")

        # Get hyperparameters for the specified profile
        hyperparams = get_hyperparameters(args.model, args.profile)

        # Add advanced features if enabled
        if args.augmentation:
            logger.info("🔄 Advanced data augmentation enabled")
            hyperparams.update(DATA_AUG_PARAMS)

        if args.early_stopping:
            logger.info("⏹️  Early stopping enabled")
            hyperparams.update(OPTIMIZATION_PARAMS)

        # Override max epochs
        hyperparams['num_epochs'] = args.max_epochs

        logger.info(f"Training configuration: {hyperparams}")

        # Import the appropriate training function
        if args.model == 'faster_rcnn':
            from src.training.train_faster_rcnn import train_faster_rcnn as train_func
        elif args.model == 'yolov8':
            from src.training.train_yolov8 import train_yolov8 as train_func
        else:
            logger.error(
                f"Advanced training not implemented for model: {args.model}")
            return False

        # Run training with advanced configuration
        success = train_func(
            dataset_name=args.dataset,
            **hyperparams
        )

        if success:
            logger.info("✅ Advanced training completed successfully!")
            logger.info(
                "📊 Check outputs directory for detailed metrics and results")
        else:
            logger.error("❌ Advanced training failed")

        return success

    except Exception as e:
        logger.error(f"Error during advanced training: {e}")
        return False


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Cattle Detection & Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s preprocess --dataset cattlebody --split-ratio 0.8
  %(prog)s train --model faster_rcnn --dataset cattlebody
  %(prog)s train --model yolov8 --dataset cattleface --epochs 50 --batch-size 16
  %(prog)s train --model faster_rcnn --dataset cattle --device 2
  %(prog)s train --model yolov8 --dataset cattleface --device cuda:1
  %(prog)s evaluate --model faster_rcnn --dataset cattleface
  %(prog)s info --list-models
  %(prog)s info --list-devices
  %(prog)s info --show-structure
        """
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('-m', '--model', required=True, choices=list(TRAINING_CONFIGS.keys()),
                              help='Model to train')

    # Dataset specification - either by name or path
    dataset_group = train_parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('-d', '--dataset', choices=list(get_available_datasets().keys()),
                               help='Dataset to use (by name)')
    dataset_group.add_argument('--dataset-path', type=str,
                               help='Direct path to dataset directory (robust mode)')

    train_parser.add_argument('-e', '--epochs', type=int,
                              help='Number of training epochs')
    train_parser.add_argument('-b', '--batch-size', type=int,
                              help='Batch size for training')
    train_parser.add_argument(
        '-lr', '--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--device', type=str, default='auto',
                              help='Device to use for training (cpu, cuda, auto, or GPU ID like 0, 1, 2, cuda:0, cuda:1, etc.)')
    train_parser.add_argument('--num-classes', type=int,
                              help='Override auto-detected number of classes (for testing)')
    train_parser.add_argument('--validate-dataset', action='store_true',
                              help='Validate dataset before training')

    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('-m', '--model', required=True, choices=list(TRAINING_CONFIGS.keys()),
                             help='Model to evaluate')
    eval_parser.add_argument('-d', '--dataset', required=True, choices=list(get_available_datasets().keys()),
                             help='Dataset to evaluate on')
    eval_parser.add_argument('-b', '--batch-size', type=int, default=4,
                             help='Batch size for evaluation')
    eval_parser.add_argument('-t', '--score-threshold', type=float, default=0.5,
                             help='Minimum confidence score threshold')
    eval_parser.add_argument('-p', '--model-path', type=str,
                             help='Path to specific model checkpoint (overrides default)')
    eval_parser.add_argument('-o', '--output-dir', type=str,
                             help='Directory to save evaluation results')

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess', help='Preprocess raw datasets for training')
    preprocess_parser.add_argument('-d', '--dataset', required=True, choices=list(get_available_datasets().keys()),
                                   help='Dataset to preprocess')
    preprocess_parser.add_argument('-s', '--split-ratio', type=float, default=0.8,
                                   help='Train/validation split ratio (default: 0.8)')
    preprocess_parser.add_argument('-f', '--force', action='store_true',
                                   help='Force reprocessing even if output exists')

    # Visualize command
    viz_parser = subparsers.add_parser(
        'visualize', help='Visualize model results')

    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Run debug tests')

    # Dataset specification (mutually exclusive options)
    debug_dataset_group = debug_parser.add_mutually_exclusive_group(
        required=True)
    debug_dataset_group.add_argument('-d', '--dataset', choices=list(get_available_datasets().keys()),
                                     help='Dataset to use (by name)')
    debug_dataset_group.add_argument('--dataset-path',
                                     help='Direct path to dataset directory (robust mode)')

    # Debug options
    debug_parser.add_argument('--validate-dataset', action='store_true',
                              help='Validate dataset before debugging')
    debug_parser.add_argument('--num-classes', type=int,
                              help='Override auto-detected number of classes (for testing)')
    debug_parser.add_argument('--sample-size', type=int, default=10,
                              help='Number of samples to analyze for debugging')

    # Hyperparameter optimization command
    hyperparam_parser = subparsers.add_parser(
        'optimize', help='Run hyperparameter optimization')
    hyperparam_parser.add_argument('-m', '--model', required=True, choices=list(TRAINING_CONFIGS.keys()),
                                   help='Model to optimize')
    hyperparam_parser.add_argument('-d', '--dataset', required=True, choices=list(get_available_datasets().keys()),
                                   help='Dataset to use for optimization')
    hyperparam_parser.add_argument('-pr', '--profile', choices=['default', 'high_precision', 'fast_training'],
                                   default='default', help='Hyperparameter profile to use')
    hyperparam_parser.add_argument('-tr', '--trials', type=int, default=10,
                                   help='Number of optimization trials')
    hyperparam_parser.add_argument('-me', '--max-epochs', type=int, default=50,
                                   help='Maximum epochs per trial')

    # Advanced training command
    advanced_parser = subparsers.add_parser(
        'train-advanced', help='Advanced training with optimization')
    advanced_parser.add_argument('-m', '--model', required=True, choices=list(TRAINING_CONFIGS.keys()),
                                 help='Model to train')
    advanced_parser.add_argument('-d', '--dataset', required=True, choices=list(get_available_datasets().keys()),
                                 help='Dataset to use')
    advanced_parser.add_argument('-pr', '--profile', choices=['default', 'high_precision', 'fast_training'],
                                 default='default', help='Training profile')
    advanced_parser.add_argument('-a', '--augmentation', action='store_true',
                                 help='Enable advanced data augmentation')
    advanced_parser.add_argument('-es', '--early-stopping', action='store_true',
                                 help='Enable early stopping')
    advanced_parser.add_argument('-me', '--max-epochs', type=int, default=200,
                                 help='Maximum training epochs')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.add_argument(
        '-l', '--list-models', action='store_true', help='List available models and datasets')
    info_parser.add_argument(
        '-d', '--list-devices', action='store_true', help='List available devices (GPUs/CPU)')
    info_parser.add_argument(
        '-s', '--show-structure', action='store_true', help='Show project structure')

    # Cleanup command
    cleanup_parser = subparsers.add_parser(
        'cleanup', help='Clean up old metric files')
    cleanup_parser.add_argument('--dir', type=str,
                                help='Specific metrics directory to clean')
    cleanup_parser.add_argument('--all', action='store_true',
                                help='Clean all metrics directories')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                                help='Show what would be deleted without deleting')
    cleanup_parser.add_argument('-f', '--force', action='store_true',
                                help='Skip confirmation prompts')

    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 0

    logger.info(f"Cattle Detection System - Command: {args.command}")

    try:
        if args.command == 'train':
            success = train_model(args, logger)
            return 0 if success else 1

        elif args.command == 'evaluate':
            success = evaluate_model(args, logger)
            return 0 if success else 1

        elif args.command == 'preprocess':
            success = preprocess_dataset(args, logger)
            return 0 if success else 1

        elif args.command == 'visualize':
            success = visualize_results(args, logger)
            return 0 if success else 1

        elif args.command == 'debug':
            success = run_debug(args, logger)
            return 0 if success else 1

        elif args.command == 'optimize':
            success = run_hyperparameter_optimization(args, logger)
            return 0 if success else 1

        elif args.command == 'train-advanced':
            success = run_advanced_training(args, logger)
            return 0 if success else 1

        elif args.command == 'info':
            show_info(args, logger)
            return 0

        elif args.command == 'cleanup':
            success = cleanup_metrics(args, logger)
            return 0 if success else 1

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

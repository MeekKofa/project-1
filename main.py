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
    
    # Run model evaluation
    python main.py evaluate --model faster_rcnn --dataset cattleface
    
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

    # Create outputs directory if it doesn't exist
    create_output_dir(OUTPUT_LOGS_DIR)

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
        "cattleface": "Cattle face detection dataset"
    }


def get_available_models() -> Dict[str, str]:
    """Get available models from training configs."""
    return {key: config["description"] for key, config in TRAINING_CONFIGS.items()}


def train_model(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Train a model with the specified configuration."""
    model_name = args.model
    dataset = args.dataset

    if model_name not in TRAINING_CONFIGS:
        logger.error(
            f"Unknown model: {model_name}. Available models: {list(TRAINING_CONFIGS.keys())}")
        return False

    if dataset not in get_available_datasets():
        logger.error(
            f"Unknown dataset: {dataset}. Available datasets: {list(get_available_datasets().keys())}")
        return False

    config = TRAINING_CONFIGS[model_name]
    logger.info(f"Starting {config['name']} training on {dataset} dataset...")
    logger.info(f"Description: {config['description']}")

    # Create systematic output directories: outputs/{dataset}/{model}/{type}/
    models_dir = get_systematic_output_dir(dataset, model_name, "models")
    logs_dir = get_systematic_output_dir(dataset, model_name, "logs")
    results_dir = get_systematic_output_dir(dataset, model_name, "results")
    images_dir = get_systematic_output_dir(dataset, model_name, "images")
    metrics_dir = get_systematic_output_dir(dataset, model_name, "metrics")
    checkpoints_dir = get_systematic_output_dir(
        dataset, model_name, "checkpoints")

    logger.info(f"Output directory: {models_dir}")
    logger.info(f"Logs directory: {logs_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Metrics directory: {metrics_dir}")

    # Create log file path
    log_file = logs_dir / f"{model_name}_{dataset}.log"

    try:
        # Import and run the training module
        module = importlib.import_module(config["module"])

        # Prepare training arguments
        train_args = {
            'dataset': dataset,
            'output_dir': str(models_dir),
            'log_file': str(log_file),
            'metrics_dir': str(metrics_dir),
            'results_dir': str(results_dir),
            'images_dir': str(images_dir),
            'checkpoints_dir': str(checkpoints_dir)
        }

        # Add optional arguments if provided
        if hasattr(args, 'epochs') and args.epochs:
            train_args['epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            train_args['batch_size'] = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            train_args['learning_rate'] = args.learning_rate
        if hasattr(args, 'device') and args.device:
            train_args['device'] = args.device

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
    """Run debug tests."""
    try:
        from run_debug_sample import main as debug_main
        debug_main()
        logger.info("✅ Debug tests completed!")
        return True
    except ImportError as e:
        logger.error(f"Failed to import debug module: {e}")
        return False
    except Exception as e:
        logger.error(f"Debug tests failed with error: {e}")
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
  %(prog)s evaluate --model faster_rcnn --dataset cattleface
  %(prog)s info --list-models
  %(prog)s info --show-structure
        """
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', required=True, choices=list(TRAINING_CONFIGS.keys()),
                              help='Model to train')
    train_parser.add_argument('--dataset', required=True, choices=list(get_available_datasets().keys()),
                              help='Dataset to use')
    train_parser.add_argument('--epochs', type=int,
                              help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int,
                              help='Batch size for training')
    train_parser.add_argument(
        '--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                              help='Device to use for training')

    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', required=True, choices=list(TRAINING_CONFIGS.keys()),
                             help='Model to evaluate')
    eval_parser.add_argument('--dataset', required=True, choices=list(get_available_datasets().keys()),
                             help='Dataset to evaluate on')
    eval_parser.add_argument('--batch-size', type=int, default=4,
                             help='Batch size for evaluation')
    eval_parser.add_argument('--score-threshold', type=float, default=0.5,
                             help='Minimum confidence score threshold')
    eval_parser.add_argument('--model-path', type=str,
                             help='Path to specific model checkpoint (overrides default)')
    eval_parser.add_argument('--output-dir', type=str,
                             help='Directory to save evaluation results')

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess', help='Preprocess raw datasets for training')
    preprocess_parser.add_argument('--dataset', required=True, choices=list(get_available_datasets().keys()),
                                   help='Dataset to preprocess')
    preprocess_parser.add_argument('--split-ratio', type=float, default=0.8,
                                   help='Train/validation split ratio (default: 0.8)')
    preprocess_parser.add_argument('--force', action='store_true',
                                   help='Force reprocessing even if output exists')

    # Visualize command
    viz_parser = subparsers.add_parser(
        'visualize', help='Visualize model results')

    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Run debug tests')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.add_argument(
        '--list-models', action='store_true', help='List available models and datasets')
    info_parser.add_argument(
        '--show-structure', action='store_true', help='Show project structure')

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

        elif args.command == 'info':
            show_info(args, logger)
            return 0

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

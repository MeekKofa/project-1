"""
Centralized argument parser for all CLI commands.
Supports train, eval, and preprocess commands.
"""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """
    Create main argument parser with subcommands.

    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Cattle Detection System - Train and evaluate object detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Command to execute',
        required=True
    )

    # ============================================
    # TRAIN COMMAND
    # ============================================
    train_parser = subparsers.add_parser(
        'train',
        help='Train a detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    train_parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        choices=['faster_rcnn', 'yolov8_resnet', 'yolov8_csp'],
        help='Model architecture to train'
    )

    train_parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., cattle, cattlebody, cattleface)'
    )

    # Training hyperparameters
    train_parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )

    train_parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=None,
        help='Batch size for training (overrides config)'
    )

    train_parser.add_argument(
        '-lr', '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    train_parser.add_argument(
        '--img-size',
        type=int,
        default=None,
        help='Input image size (overrides config)'
    )

    # Optimizer settings
    train_parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'adamw', 'sgd'],
        default=None,
        help='Optimizer type (overrides config)'
    )

    train_parser.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help='Weight decay for optimizer (overrides config)'
    )

    # Device and performance
    train_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to use for training'
    )

    train_parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )

    # Config file
    train_parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Path to custom config YAML file (overrides default config.yaml)'
    )

    # Checkpoint options
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    train_parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=None,
        help='Save checkpoint every N epochs (overrides config)'
    )

    # Logging and visualization
    train_parser.add_argument(
        '--log-freq',
        type=int,
        default=10,
        help='Log training stats every N iterations'
    )

    train_parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization generation'
    )

    # Advanced options
    train_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    train_parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Use mixed precision training (FP16)'
    )

    # ============================================
    # EVAL COMMAND
    # ============================================
    eval_parser = subparsers.add_parser(
        'eval',
        help='Evaluate a trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    eval_parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        choices=['faster_rcnn', 'yolov8_resnet', 'yolov8_csp'],
        help='Model architecture to evaluate'
    )

    eval_parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        help='Dataset name to evaluate on'
    )

    eval_parser.add_argument(
        '-p', '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )

    eval_parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )

    eval_parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )

    eval_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to use for evaluation'
    )

    eval_parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation metrics'
    )

    eval_parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for predictions'
    )

    eval_parser.add_argument(
        '--save-predictions',
        dest='save_predictions',
        action='store_true',
        help='Enable saving consolidated predictions'
    )

    eval_parser.add_argument(
        '--no-save-predictions',
        dest='save_predictions',
        action='store_false',
        help='Disable saving consolidated predictions'
    )

    eval_parser.set_defaults(save_predictions=None)

    eval_parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Optional custom name for the evaluation run'
    )

    # ============================================
    # PREPROCESS COMMAND
    # ============================================
    prep_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    prep_parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        help='Dataset name to preprocess'
    )

    prep_parser.add_argument(
        '-s', '--train-split',
        type=float,
        default=0.8,
        help='Fraction of data for training (rest split between val/test)'
    )

    prep_parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Fraction of data for validation'
    )

    prep_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force reprocessing even if processed data exists'
    )

    prep_parser.add_argument(
        '--format',
        type=str,
        choices=['yolo', 'coco', 'pascal'],
        default='yolo',
        help='Source dataset format'
    )

    prep_parser.add_argument(
        '--output-dir',
        type=str,
        default='processed_data',
        help='Output directory for processed data'
    )

    return parser


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace: Parsed arguments
    """
    parser = create_parser()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Test argument parser
    parser = create_parser()
    parser.print_help()

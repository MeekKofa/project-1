"""
Main Entry Point - Cattle Detection System
Coordinates all commands: train, eval, preprocess
"""

from src.config.manager import get_config
from src.cli.args import parse_args
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point for the application."""

    # Parse command-line arguments
    args = parse_args()

    # Convert args to dictionary
    args_dict = vars(args)
    command = args_dict.pop('command')

    # Merge configuration (CLI > YAML > Defaults)
    config = get_config(args_dict)

    print("\n" + "="*70)
    print("CATTLE DETECTION SYSTEM")
    print("="*70)
    print(f"Command: {command}")
    print(f"Model: {config.get('model', 'N/A')}")
    print(f"Dataset: {config.get('dataset', 'N/A')}")
    print("="*70 + "\n")

    # Route to appropriate command handler
    if command == 'train':
        handle_train(config)
    elif command == 'eval':
        handle_eval(config)
    elif command == 'preprocess':
        handle_preprocess(config)
    else:
        print(f"Error: Unknown command '{command}'")
        sys.exit(1)


def handle_train(config: dict):
    """
    Handle train command.

    Args:
        config: Merged configuration dictionary
    """
    from src.training.orchestrator import train_model

    try:
        results = train_model(config)
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Best metric: {results['best_metric']:.4f}")
        print(
            f"Output directory: outputs/{config['dataset']}/{config['model']}")
        print("="*70 + "\n")

    except Exception as e:
        print("\n" + "="*70)
        print("✗ TRAINING FAILED!")
        print("="*70)
        print(f"Error: {e}")
        print("="*70 + "\n")
        raise


def handle_eval(config: dict):
    """
    Handle eval command.

    Args:
        config: Merged configuration dictionary
    """
    print("\n" + "="*70)
    print("EVALUATION MODE")
    print("="*70)
    print("Note: Evaluation pipeline is not yet integrated into train.py")
    print("Evaluation happens automatically during training.")
    print("Check outputs/{dataset}/{model}/metrics/ for results.")
    print("="*70 + "\n")

    print("To manually evaluate a checkpoint:")
    print("  1. Load the model from checkpoints/best.pth")
    print("  2. Run inference on validation set")
    print("  3. Calculate metrics (mAP, precision, recall)")
    print("\nSee archived evaluation code in archive/old_evaluation/")


def handle_preprocess(config: dict):
    """
    Handle preprocess command.

    Args:
        config: Merged configuration dictionary
    """
    print("\n" + "="*70)
    print("PREPROCESSING MODE")
    print("="*70)
    print("Note: Preprocessing pipeline is not yet integrated into train.py")
    print("Preprocessing happens automatically during data loading.")
    print("="*70 + "\n")

    print("Current preprocessing features (automatic):")
    print("  ✓ Image resizing and padding")
    print("  ✓ Normalization")
    print("  ✓ Data augmentation (flip, color jitter)")
    print("  ✓ YOLO format parsing")
    print("\nFor custom preprocessing, see:")
    print("  - src/loaders/transforms.py")
    print("  - archive/old_analysis_scripts/ (dataset analysis)")


if __name__ == '__main__':
    main()

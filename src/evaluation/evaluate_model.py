#!/usr/bin/env python3
"""
Standalone evaluation script for cattle detection models.
Can be used to evaluate trained models on validation/test datasets.
"""

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.models.faster_rcnn import create_cattle_detection_model
from src.processing.dataset import CattleDataset, collate_fn
from src.evaluation.metrics import DetectionMetrics, evaluate_model
import argparse
import torch
from pathlib import Path
import logging
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device, num_classes: int = 2):
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to the model checkpoint
        device: Device to load model on
        num_classes: Number of classes

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")

    # Create model architecture
    model = create_cattle_detection_model(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def create_dataloader(images_dir: str, labels_dir: str, batch_size: int = 4):
    """
    Create a DataLoader for evaluation.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        batch_size: Batch size for evaluation

    Returns:
        DataLoader
    """
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = CattleDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    logger.info(f"Created dataset with {len(dataset)} images")
    return dataloader


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate cattle detection models')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name (e.g., cattlebody, cattleface)')
    parser.add_argument('-i', '--images', type=str,
                        help='Path to images directory (overrides dataset)')
    parser.add_argument('-l', '--labels', type=str,
                        help='Path to labels directory (overrides dataset)')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('-t', '--score-threshold', type=float, default=0.5,
                        help='Minimum confidence score threshold')
    parser.add_argument('-o', '--output-dir', type=str, default='./outputs/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Setup paths
    if args.images and args.labels:
        images_dir = args.images
        labels_dir = args.labels
    else:
        # Use dataset name to construct paths
        processed_data_dir = Path(project_root) / \
            'processed_data' / args.dataset
        images_dir = processed_data_dir / 'val' / 'images'
        labels_dir = processed_data_dir / 'val' / 'labels'

        if not images_dir.exists():
            images_dir = processed_data_dir / 'test' / 'images'
            labels_dir = processed_data_dir / 'test' / 'labels'

    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Labels directory: {labels_dir}")

    # Validate paths
    if not Path(images_dir).exists():
        logger.error(f"Images directory does not exist: {images_dir}")
        return 1

    if not Path(labels_dir).exists():
        logger.error(f"Labels directory does not exist: {labels_dir}")
        return 1

    if not Path(args.model).exists():
        logger.error(f"Model file does not exist: {args.model}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        model = load_model(args.model, device)

        # Create dataloader
        dataloader = create_dataloader(images_dir, labels_dir, args.batch_size)

        if len(dataloader) == 0:
            logger.error("No images found in the dataset")
            return 1

        # Run evaluation
        logger.info("Starting comprehensive evaluation...")
        results, metrics_obj = evaluate_model(
            model, dataloader, device,
            score_threshold=args.score_threshold,
            num_classes=2
        )

        # Print results
        metrics_obj.print_metrics(results)

        # Save detailed results
        results_file = output_dir / f"evaluation_results_{args.dataset}.json"
        metrics_obj.save_metrics(results, str(results_file))

        # Save summary
        summary_file = output_dir / f"evaluation_summary_{args.dataset}.txt"
        with open(summary_file, 'w') as f:
            f.write("CATTLE DETECTION MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Images: {images_dir}\n")
            f.write(f"Labels: {labels_dir}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Score Threshold: {args.score_threshold}\n\n")

            summary = results.get('summary', {})
            f.write("KEY METRICS:\n")
            f.write(f"  mAP@0.5      : {summary.get('mAP@0.5', 0):.4f}\n")
            f.write(f"  mAP@0.75     : {summary.get('mAP@0.75', 0):.4f}\n")
            f.write(f"  mAP@0.5:0.95 : {summary.get('mAP@0.5:0.95', 0):.4f}\n")
            f.write(
                f"  Precision@0.5: {summary.get('precision@0.5', 0):.4f}\n")
            f.write(f"  Recall@0.5   : {summary.get('recall@0.5', 0):.4f}\n")
            f.write(f"  F1@0.5       : {summary.get('f1@0.5', 0):.4f}\n\n")

            f.write(f"DATASET STATISTICS:\n")
            f.write(f"  Images evaluated: {len(dataloader.dataset)}\n")
            f.write(
                f"  Total predictions: {summary.get('num_predictions', 0)}\n")
            f.write(
                f"  Total ground truths: {summary.get('num_ground_truths', 0)}\n")

        logger.info(f"✅ Evaluation completed successfully!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"📄 Summary saved to: {summary_file}")

        return 0

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

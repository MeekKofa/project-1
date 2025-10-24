"""
Validation script to verify training improvements.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

from ..evaluation.metrics import DetectionMetricsSimple
from ..utils.visualization import visualize_detections

logger = logging.getLogger(__name__)

class ValidationManager:
    """
    Manages validation of trained models and compares performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        output_dir: str,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.device = device
        self.metrics = DetectionMetricsSimple(num_classes=2)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @torch.no_grad()
    def validate_model(self) -> Dict[str, Any]:
        """Run full validation and generate report."""
        self.model.eval()
        self.metrics.reset()
        
        # Track per-image metrics
        image_metrics = []
        confidence_scores = []
        box_sizes = []
        
        logger.info("Running validation...")
        
        for images, targets in tqdm(self.val_loader):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Get predictions
            predictions = self.model(images)
            
            # Update metrics
            self.metrics.update(predictions, targets)
            
            # Collect per-image statistics
            for pred, target in zip(predictions, targets):
                # Confidence scores
                if len(pred['scores']) > 0:
                    confidence_scores.extend(pred['scores'].cpu().numpy())
                    
                # Box sizes
                if len(pred['boxes']) > 0:
                    boxes = pred['boxes'].cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    box_sizes.extend(areas)
                    
                # Per-image metrics
                img_metrics = {
                    'num_predictions': len(pred['boxes']),
                    'num_targets': len(target['boxes']),
                    'mean_confidence': pred['scores'].mean().item() if len(pred['scores']) > 0 else 0,
                    'max_confidence': pred['scores'].max().item() if len(pred['scores']) > 0 else 0
                }
                image_metrics.append(img_metrics)
        
        # Compute overall metrics
        results = {
            'mAP50': self.metrics.compute_map(),
            'precision': self.metrics.compute_precision(),
            'recall': self.metrics.compute_recall(),
            'avg_predictions_per_image': np.mean([m['num_predictions'] for m in image_metrics]),
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_distribution': {
                'min': np.min(confidence_scores) if confidence_scores else 0,
                'max': np.max(confidence_scores) if confidence_scores else 0,
                'mean': np.mean(confidence_scores) if confidence_scores else 0,
                'std': np.std(confidence_scores) if confidence_scores else 0
            },
            'box_size_statistics': {
                'min': np.min(box_sizes) if box_sizes else 0,
                'max': np.max(box_sizes) if box_sizes else 0,
                'mean': np.mean(box_sizes) if box_sizes else 0,
                'std': np.std(box_sizes) if box_sizes else 0
            }
        }
        
        # Generate visualizations
        self._plot_confidence_distribution(confidence_scores)
        self._plot_box_size_distribution(box_sizes)
        
        return results
        
    def _plot_confidence_distribution(self, scores: List[float]):
        """Plot distribution of confidence scores."""
        if not scores:
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.75)
        plt.title('Distribution of Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'confidence_distribution.png')
        plt.close()
        
    def _plot_box_size_distribution(self, sizes: List[float]):
        """Plot distribution of bounding box sizes."""
        if not sizes:
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(sizes, bins=50, alpha=0.75)
        plt.title('Distribution of Bounding Box Sizes')
        plt.xlabel('Box Area (pixels²)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'box_size_distribution.png')
        plt.close()
        
    def visualize_sample_predictions(self, num_samples: int = 10):
        """Visualize sample predictions."""
        self.model.eval()
        
        # Get random samples
        indices = np.random.choice(len(self.val_loader.dataset), num_samples)
        
        for idx in indices:
            image, target = self.val_loader.dataset[idx]
            
            # Get prediction
            with torch.no_grad():
                prediction = self.model([image.to(self.device)])[0]
                
            # Visualize
            fig = visualize_detections(
                image.cpu(),
                prediction['boxes'].cpu(),
                prediction['scores'].cpu(),
                target['boxes'].cpu(),
                threshold=0.5
            )
            
            # Save visualization
            plt.savefig(self.output_dir / f'sample_prediction_{idx}.png')
            plt.close()
            
    def generate_report(self, results: Dict[str, Any]):
        """Generate validation report."""
        report = [
            "# Validation Report\n",
            "## Model Performance Metrics\n",
            f"- mAP50: {results['mAP50']:.4f}",
            f"- Precision: {results['precision']:.4f}",
            f"- Recall: {results['recall']:.4f}\n",
            "## Detection Statistics\n",
            f"- Average predictions per image: {results['avg_predictions_per_image']:.2f}",
            f"- Average confidence score: {results['avg_confidence']:.4f}\n",
            "## Confidence Score Distribution\n",
            "```",
            f"Min: {results['confidence_distribution']['min']:.4f}",
            f"Max: {results['confidence_distribution']['max']:.4f}",
            f"Mean: {results['confidence_distribution']['mean']:.4f}",
            f"Std: {results['confidence_distribution']['std']:.4f}",
            "```\n",
            "## Bounding Box Statistics\n",
            "```",
            f"Min area: {results['box_size_statistics']['min']:.2f}",
            f"Max area: {results['box_size_statistics']['max']:.2f}",
            f"Mean area: {results['box_size_statistics']['mean']:.2f}",
            f"Std area: {results['box_size_statistics']['std']:.2f}",
            "```\n",
            "## Visualizations\n",
            "1. Confidence score distribution: confidence_distribution.png",
            "2. Box size distribution: box_size_distribution.png",
            "3. Sample predictions: sample_prediction_*.png"
        ]
        
        # Save report
        with open(self.output_dir / 'validation_report.md', 'w') as f:
            f.write('\n'.join(report))
            
        # Save raw results
        with open(self.output_dir / 'validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

def validate_improvements(
    model: nn.Module,
    val_loader: DataLoader,
    output_dir: str,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Validate model improvements.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        output_dir: Output directory for validation results
        device: Device to run validation on
        
    Returns:
        Validation results
    """
    validator = ValidationManager(model, val_loader, output_dir, device)
    
    # Run validation
    results = validator.validate_model()
    
    # Generate visualizations
    validator.visualize_sample_predictions()
    
    # Generate report
    validator.generate_report(results)
    
    return results
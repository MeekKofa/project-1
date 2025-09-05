"""
Comprehensive metrics evaluation module for cattle detection models.
Supports mAP, precision, recall, F1-score, and other detection metrics.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import csv
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    CSV-based metrics tracker for consolidating training/evaluation metrics.
    Saves only essential metrics to CSV for trend analysis.
    """

    def __init__(self, csv_path: str):
        """Initialize metrics tracker with CSV file path."""
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV with headers if it doesn't exist
        if not self.csv_path.exists():
            self.initialize_csv()

    def initialize_csv(self):
        """Initialize CSV file with headers."""
        headers = [
            'timestamp', 'epoch', 'split', 'mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95',
            'precision@0.5', 'recall@0.5', 'f1@0.5', 'num_predictions', 'num_ground_truths',
            'score_threshold', 'model_type', 'dataset', 'notes'
        ]

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        logger.info(f"Initialized metrics CSV at {self.csv_path}")

    def log_metrics(self,
                    metrics: Dict,
                    epoch: int = None,
                    split: str = 'val',
                    model_type: str = '',
                    dataset: str = '',
                    notes: str = ''):
        """
        Log metrics to CSV file.

        Args:
            metrics: Dictionary containing metrics results
            epoch: Training epoch number (None for final evaluation)
            split: Data split (train/val/test)
            model_type: Model type (faster_rcnn, yolov8, etc.)
            dataset: Dataset name
            notes: Additional notes
        """
        summary = metrics.get('summary', {})

        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            epoch if epoch is not None else 'final',
            split,
            summary.get('mAP@0.5', 0),
            summary.get('mAP@0.75', 0),
            summary.get('mAP@0.5:0.95', 0),
            summary.get('precision@0.5', 0),
            summary.get('recall@0.5', 0),
            summary.get('f1@0.5', 0),
            summary.get('num_predictions', 0),
            summary.get('num_ground_truths', 0),
            summary.get('score_threshold', 0.5),
            model_type,
            dataset,
            notes
        ]

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        if epoch is not None:
            logger.info(f"Logged epoch {epoch} metrics to {self.csv_path}")
        else:
            logger.info(f"Logged final evaluation metrics to {self.csv_path}")

    def get_training_history(self) -> List[Dict]:
        """Get training history from CSV file."""
        if not self.csv_path.exists():
            return []

        history = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(row)

        return history

    def plot_training_curves(self, save_path: str = None):
        """Plot training curves from CSV data."""
        history = self.get_training_history()
        if not history:
            logger.warning("No training history found for plotting")
            return

        # Filter only training epochs (not 'final')
        train_history = [row for row in history if row['epoch']
                         != 'final' and row['epoch'].isdigit()]

        if not train_history:
            logger.warning("No epoch data found for plotting")
            return

        # Extract data for plotting
        epochs = [int(row['epoch']) for row in train_history]
        map50 = [float(row['mAP@0.5']) for row in train_history]
        precision = [float(row['precision@0.5']) for row in train_history]
        recall = [float(row['recall@0.5']) for row in train_history]
        f1 = [float(row['f1@0.5']) for row in train_history]

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # mAP@0.5
        ax1.plot(epochs, map50, 'b-', linewidth=2, marker='o')
        ax1.set_title('mAP@0.5 Over Training', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mAP@0.5')
        ax1.grid(True, alpha=0.3)

        # Precision
        ax2.plot(epochs, precision, 'g-', linewidth=2, marker='s')
        ax2.set_title('Precision@0.5 Over Training',
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Precision@0.5')
        ax2.grid(True, alpha=0.3)

        # Recall
        ax3.plot(epochs, recall, 'r-', linewidth=2, marker='^')
        ax3.set_title('Recall@0.5 Over Training',
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Recall@0.5')
        ax3.grid(True, alpha=0.3)

        # F1-Score
        ax4.plot(epochs, f1, 'm-', linewidth=2, marker='D')
        ax4.set_title('F1-Score@0.5 Over Training',
                      fontsize=12, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1-Score@0.5')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        else:
            plt.show()

        plt.close()


class DetectionMetrics:
    """
    Comprehensive metrics calculator for object detection tasks.
    Supports COCO-style evaluation with multiple IoU thresholds.
    """

    def __init__(self,
                 iou_thresholds: Optional[List[float]] = None,
                 score_threshold: float = 0.5,
                 num_classes: int = 2):
        """
        Initialize metrics calculator.

        Args:
            iou_thresholds: IoU thresholds for evaluation (default: [0.5:0.95:0.05])
            score_threshold: Minimum confidence score threshold
            num_classes: Number of classes (including background)
        """
        if iou_thresholds is None:
            self.iou_thresholds = [0.5 + 0.05 *
                                   i for i in range(10)]  # 0.5:0.95:0.05
        else:
            self.iou_thresholds = iou_thresholds

        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.ground_truths = []
        self.image_ids = []

    def update(self, predictions: List[Dict], targets: List[Dict], image_ids: Optional[List[int]] = None):
        """
        Update metrics with batch predictions and targets.

        Args:
            predictions: List of prediction dictionaries with 'boxes', 'scores', 'labels'
            targets: List of target dictionaries with 'boxes', 'labels'
            image_ids: Optional list of image IDs
        """
        if image_ids is None:
            image_ids = list(range(len(self.image_ids), len(
                self.image_ids) + len(predictions)))

        for pred, target, img_id in zip(predictions, targets, image_ids):
            # Filter predictions by score threshold
            if 'scores' in pred and len(pred['scores']) > 0:
                valid_mask = pred['scores'] >= self.score_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][valid_mask] if len(pred['boxes']) > 0 else torch.empty(0, 4),
                    'scores': pred['scores'][valid_mask] if len(pred['scores']) > 0 else torch.empty(0),
                    'labels': pred['labels'][valid_mask] if len(pred['labels']) > 0 else torch.empty(0, dtype=torch.long)
                }
            else:
                filtered_pred = {
                    'boxes': pred.get('boxes', torch.empty(0, 4)),
                    'scores': pred.get('scores', torch.empty(0)),
                    'labels': pred.get('labels', torch.empty(0, dtype=torch.long))
                }

            self.predictions.append(filtered_pred)
            self.ground_truths.append(target)
            self.image_ids.append(img_id)

    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes.

        Args:
            box1: Boxes of shape (N, 4) in [x1, y1, x2, y2] format
            box2: Boxes of shape (M, 4) in [x1, y1, x2, y2] format

        Returns:
            IoU matrix of shape (N, M)
        """
        if box1.numel() == 0 or box2.numel() == 0:
            return torch.zeros(box1.shape[0], box2.shape[0])

        # Expand dimensions for broadcasting
        box1 = box1.unsqueeze(1)  # (N, 1, 4)
        box2 = box2.unsqueeze(0)  # (1, M, 4)

        # Calculate intersection
        inter_x1 = torch.max(box1[..., 0], box2[..., 0])
        inter_y1 = torch.max(box1[..., 1], box2[..., 1])
        inter_x2 = torch.min(box1[..., 2], box2[..., 2])
        inter_y2 = torch.min(box1[..., 3], box2[..., 3])

        inter_area = torch.clamp(
            inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Calculate union
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union_area = area1 + area2 - inter_area

        # Compute IoU
        iou = inter_area / (union_area + 1e-7)
        return iou.squeeze()

    def compute_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """
        Compute Average Precision using the 101-point interpolation method.

        Args:
            precision: Precision values
            recall: Recall values

        Returns:
            Average Precision
        """
        # Add sentinel values
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Integrate area under curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def evaluate_class(self, class_id: int, iou_threshold: float) -> Dict[str, float]:
        """
        Evaluate metrics for a specific class and IoU threshold.

        Args:
            class_id: Class ID to evaluate
            iou_threshold: IoU threshold

        Returns:
            Dictionary with precision, recall, AP, and F1 score
        """
        # Collect all predictions and ground truths for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        all_image_ids_pred = []
        all_image_ids_gt = []

        for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
            # Filter predictions for this class
            if len(pred['labels']) > 0:
                class_mask = pred['labels'] == class_id
                if class_mask.any():
                    all_pred_boxes.extend(
                        pred['boxes'][class_mask].cpu().numpy())
                    all_pred_scores.extend(
                        pred['scores'][class_mask].cpu().numpy())
                    all_image_ids_pred.extend([i] * class_mask.sum().item())

            # Filter ground truths for this class
            if len(gt['labels']) > 0:
                gt_class_mask = gt['labels'] == class_id
                if gt_class_mask.any():
                    all_gt_boxes.extend(
                        gt['boxes'][gt_class_mask].cpu().numpy())
                    all_image_ids_gt.extend([i] * gt_class_mask.sum().item())

        if len(all_pred_boxes) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'ap': 0.0, 'f1': 0.0}

        if len(all_gt_boxes) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'ap': 0.0, 'f1': 0.0}

        try:
            # Convert to tensors (handle numpy array conversion properly)
            pred_boxes = torch.tensor(
                np.array(all_pred_boxes), dtype=torch.float32)
            pred_scores = torch.tensor(
                np.array(all_pred_scores), dtype=torch.float32)
            gt_boxes = torch.tensor(
                np.array(all_gt_boxes), dtype=torch.float32)

            # Ensure proper dimensions
            if pred_boxes.dim() != 2 or pred_boxes.shape[1] != 4:
                logger.warning(f"Invalid pred_boxes shape: {pred_boxes.shape}")
                return {'precision': 0.0, 'recall': 0.0, 'ap': 0.0, 'f1': 0.0}
            if gt_boxes.dim() != 2 or gt_boxes.shape[1] != 4:
                logger.warning(f"Invalid gt_boxes shape: {gt_boxes.shape}")
                return {'precision': 0.0, 'recall': 0.0, 'ap': 0.0, 'f1': 0.0}
            if pred_scores.dim() != 1:
                logger.warning(
                    f"Invalid pred_scores shape: {pred_scores.shape}")
                return {'precision': 0.0, 'recall': 0.0, 'ap': 0.0, 'f1': 0.0}

        except Exception as e:
            logger.error(f"Error converting to tensors: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'ap': 0.0, 'f1': 0.0}

        # Sort predictions by confidence score (descending)
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        pred_image_ids = [all_image_ids_pred[i]
                          for i in sorted_indices.cpu().numpy()]

        # Track which ground truths have been matched
        num_gt = len(all_gt_boxes)
        gt_matched = [False] * num_gt

        # Calculate TP and FP for each prediction
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))

        for pred_idx, (pred_box, pred_img_id) in enumerate(zip(pred_boxes, pred_image_ids)):
            # Find ground truths in the same image
            gt_indices = [i for i, img_id in enumerate(
                all_image_ids_gt) if img_id == pred_img_id]

            if not gt_indices:
                fp[pred_idx] = 1
                continue

            # Calculate IoU with all ground truths in this image
            gt_boxes_img = gt_boxes[gt_indices]
            if len(gt_boxes_img) == 0:
                fp[pred_idx] = 1
                continue

            ious = self.compute_iou(pred_box.unsqueeze(0), gt_boxes_img)

            if ious.numel() == 0:
                fp[pred_idx] = 1
                continue

            # Handle both scalar and vector IoU results
            if ious.dim() == 0:
                max_iou = ious
                max_idx = 0
            elif ious.dim() == 1:
                max_iou, max_idx_tensor = torch.max(ious, dim=0)
                max_idx = max_idx_tensor.item() if max_idx_tensor.numel() == 1 else 0
            else:
                max_iou, max_idx_tensor = torch.max(ious, dim=1)
                max_iou = max_iou.item() if max_iou.numel(
                ) == 1 else max_iou[0].item()
                max_idx = max_idx_tensor.item() if max_idx_tensor.numel(
                ) == 1 else max_idx_tensor[0].item()
            actual_gt_idx = gt_indices[max_idx]

            # Convert max_iou to scalar if it's a tensor
            max_iou_value = max_iou.item() if torch.is_tensor(max_iou) else max_iou

            if max_iou_value >= iou_threshold:
                if not gt_matched[actual_gt_idx]:
                    tp[pred_idx] = 1
                    gt_matched[actual_gt_idx] = True
                else:
                    fp[pred_idx] = 1  # Ground truth already matched
            else:
                fp[pred_idx] = 1

        # Calculate cumulative precision and recall
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)

        precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-7)
        recall = cumsum_tp / (num_gt + 1e-7)

        # Calculate AP
        ap = self.compute_ap(precision, recall)

        # Calculate overall precision, recall, and F1
        final_precision = precision[-1] if len(precision) > 0 else 0.0
        final_recall = recall[-1] if len(recall) > 0 else 0.0
        f1 = 2 * final_precision * final_recall / \
            (final_precision + final_recall + 1e-7)

        return {
            'precision': final_precision,
            'recall': final_recall,
            'ap': ap,
            'f1': f1
        }

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive detection metrics.

        Returns:
            Dictionary with detailed metrics including mAP, class-wise metrics, etc.
        """
        if not self.predictions:
            logger.warning("No predictions available for metrics computation")
            return {}

        results = {
            'per_class_metrics': {},
            'iou_metrics': {},
            'summary': {}
        }

        # Evaluate each class at each IoU threshold
        for class_id in range(1, self.num_classes):  # Skip background class (0)
            results['per_class_metrics'][class_id] = {}

            for iou_thresh in self.iou_thresholds:
                class_metrics = self.evaluate_class(class_id, iou_thresh)
                results['per_class_metrics'][class_id][iou_thresh] = class_metrics

        # Calculate mAP metrics
        all_aps_50 = []  # AP@0.5
        all_aps_75 = []  # AP@0.75
        all_aps_50_95 = []  # AP@0.5:0.95

        for class_id in range(1, self.num_classes):
            if class_id in results['per_class_metrics']:
                # AP@0.5
                if 0.5 in results['per_class_metrics'][class_id]:
                    all_aps_50.append(
                        results['per_class_metrics'][class_id][0.5]['ap'])

                # AP@0.75
                if 0.75 in results['per_class_metrics'][class_id]:
                    all_aps_75.append(
                        results['per_class_metrics'][class_id][0.75]['ap'])

                # AP@0.5:0.95 (average over IoU thresholds)
                class_aps = [results['per_class_metrics'][class_id][iou]['ap']
                             for iou in self.iou_thresholds
                             if iou in results['per_class_metrics'][class_id]]
                if class_aps:
                    all_aps_50_95.append(np.mean(class_aps))

        # Summary metrics
        results['summary'] = {
            'mAP@0.5': np.mean(all_aps_50) if all_aps_50 else 0.0,
            'mAP@0.75': np.mean(all_aps_75) if all_aps_75 else 0.0,
            'mAP@0.5:0.95': np.mean(all_aps_50_95) if all_aps_50_95 else 0.0,
            'num_predictions': len(self.predictions),
            'num_ground_truths': sum(len(gt['labels']) for gt in self.ground_truths),
            'score_threshold': self.score_threshold
        }

        # Add class-wise summary at IoU 0.5
        if all_aps_50:
            results['summary']['precision@0.5'] = np.mean([
                results['per_class_metrics'][class_id][0.5]['precision']
                for class_id in range(1, self.num_classes)
                if class_id in results['per_class_metrics'] and 0.5 in results['per_class_metrics'][class_id]
            ])
            results['summary']['recall@0.5'] = np.mean([
                results['per_class_metrics'][class_id][0.5]['recall']
                for class_id in range(1, self.num_classes)
                if class_id in results['per_class_metrics'] and 0.5 in results['per_class_metrics'][class_id]
            ])
            results['summary']['f1@0.5'] = np.mean([
                results['per_class_metrics'][class_id][0.5]['f1']
                for class_id in range(1, self.num_classes)
                if class_id in results['per_class_metrics'] and 0.5 in results['per_class_metrics'][class_id]
            ])

        return results

    def print_metrics(self, results: Optional[Dict] = None):
        """
        Print formatted metrics summary.

        Args:
            results: Metrics results (if None, will compute them)
        """
        if results is None:
            results = self.compute_metrics()

        if not results:
            print("No metrics to display")
            return

        summary = results.get('summary', {})

        print("\n" + "="*60)
        print("DETECTION METRICS SUMMARY")
        print("="*60)

        print(f"📊 Dataset Statistics:")
        print(f"   • Images evaluated: {len(self.predictions)}")
        print(f"   • Total predictions: {summary.get('num_predictions', 0)}")
        print(
            f"   • Total ground truths: {summary.get('num_ground_truths', 0)}")
        print(
            f"   • Score threshold: {summary.get('score_threshold', self.score_threshold):.2f}")

        print(f"\n🎯 Average Precision (AP) Metrics:")
        print(f"   • mAP@0.5      : {summary.get('mAP@0.5', 0):.4f}")
        print(f"   • mAP@0.75     : {summary.get('mAP@0.75', 0):.4f}")
        print(f"   • mAP@0.5:0.95 : {summary.get('mAP@0.5:0.95', 0):.4f}")

        print(f"\n📈 Classification Metrics @ IoU 0.5:")
        print(f"   • Precision    : {summary.get('precision@0.5', 0):.4f}")
        print(f"   • Recall       : {summary.get('recall@0.5', 0):.4f}")
        print(f"   • F1-Score     : {summary.get('f1@0.5', 0):.4f}")

        # Per-class metrics
        per_class = results.get('per_class_metrics', {})
        if per_class:
            print(f"\n🔍 Per-Class Metrics @ IoU 0.5:")
            for class_id, class_metrics in per_class.items():
                if 0.5 in class_metrics:
                    metrics = class_metrics[0.5]
                    class_name = f"Class {class_id}" if class_id != 1 else "Cattle"
                    print(f"   • {class_name:8s}: AP={metrics['ap']:.4f}, "
                          f"P={metrics['precision']:.4f}, "
                          f"R={metrics['recall']:.4f}, "
                          f"F1={metrics['f1']:.4f}")

        print("="*60)

    def save_metrics(self, results: Dict, save_path: str, epoch: int = None,
                     model_type: str = '', dataset: str = '', save_individual: bool = False):
        """
        Save metrics results with CSV tracking and enhanced reporting.

        Args:
            results: Metrics results
            save_path: Base path to save the metrics
            epoch: Training epoch (None for final evaluation)
            model_type: Model type for CSV tracking
            dataset: Dataset name for CSV tracking
            save_individual: Whether to save individual epoch files (default: False)
        """
        import json

        base_path = Path(save_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup CSV tracker
        csv_path = base_path.parent / 'training_metrics.csv'
        tracker = MetricsTracker(str(csv_path))

        # Log to CSV (always)
        tracker.log_metrics(
            results,
            epoch=epoch,
            model_type=model_type,
            dataset=dataset,
            notes=f"Evaluation at epoch {epoch}" if epoch else "Final evaluation"
        )

        # Only save individual files if requested or for final evaluation
        if save_individual or epoch is None:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj

            serializable_results = convert_numpy(results)

            # For final evaluation, use enhanced naming
            if epoch is None:
                json_path = str(base_path).replace('.json', '_final.json')
                txt_path = str(base_path).replace('.json', '_final_report.txt')
            else:
                json_path = str(base_path)
                txt_path = str(base_path).replace('.json', '_summary.txt')

            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"Metrics saved to {json_path}")

            # Save enhanced text summary
            self.save_enhanced_summary(
                results, txt_path, epoch, model_type, dataset)

        # Generate training curves if we have epoch data
        if epoch is not None:
            curves_path = base_path.parent / 'training_curves.png'
            try:
                tracker.plot_training_curves(str(curves_path))
            except Exception as e:
                logger.warning(f"Could not generate training curves: {e}")

    def save_enhanced_summary(self, results: Dict, save_path: str, epoch: int = None,
                              model_type: str = '', dataset: str = ''):
        """
        Save an enhanced human-readable text summary of metrics.

        Args:
            results: Metrics results
            save_path: Path to save the text summary
            epoch: Training epoch (None for final evaluation)
            model_type: Model type
            dataset: Dataset name
        """
        from datetime import datetime
        import platform

        summary = results.get('summary', {})
        per_class = results.get('per_class_metrics', {})

        with open(save_path, 'w') as f:
            # Enhanced Header
            f.write("🐄 CATTLE DETECTION MODEL - EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            f.write(
                f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"💻 System: {platform.system()} {platform.release()}\n")
            f.write(f"🐍 Python: {platform.python_version()}\n")
            if model_type:
                f.write(f"🧠 Model: {model_type}\n")
            if dataset:
                f.write(f"📊 Dataset: {dataset}\n")
            if epoch is not None:
                f.write(f"🔄 Epoch: {epoch}\n")
            else:
                f.write("✅ Final Evaluation\n")
            f.write("\n")

            # Dataset Statistics
            f.write("📊 DATASET STATISTICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Images evaluated: {len(self.predictions):,}\n")
            f.write(
                f"Total predictions: {summary.get('num_predictions', 0):,}\n")
            f.write(
                f"Total ground truths: {summary.get('num_ground_truths', 0):,}\n")
            f.write(
                f"Score threshold: {summary.get('score_threshold', self.score_threshold):.2f}\n\n")

            # Key Performance Metrics
            f.write("🎯 KEY PERFORMANCE METRICS:\n")
            f.write("-" * 60 + "\n")
            map50 = summary.get('mAP@0.5', 0)
            map75 = summary.get('mAP@0.75', 0)
            map_coco = summary.get('mAP@0.5:0.95', 0)
            precision = summary.get('precision@0.5', 0)
            recall = summary.get('recall@0.5', 0)
            f1 = summary.get('f1@0.5', 0)

            f.write(f"mAP@0.5       : {map50:.4f} ({map50*100:.2f}%)\n")
            f.write(f"mAP@0.75      : {map75:.4f} ({map75*100:.2f}%)\n")
            f.write(f"mAP@0.5:0.95  : {map_coco:.4f} ({map_coco*100:.2f}%)\n")
            f.write(
                f"Precision@0.5 : {precision:.4f} ({precision*100:.2f}%)\n")
            f.write(f"Recall@0.5    : {recall:.4f} ({recall*100:.2f}%)\n")
            f.write(f"F1-Score@0.5  : {f1:.4f} ({f1*100:.2f}%)\n\n")

            # Enhanced Performance Interpretation
            f.write("🔍 PERFORMANCE INTERPRETATION:\n")
            f.write("-" * 60 + "\n")

            # Overall performance rating
            if map50 >= 0.85:
                f.write("🟢 EXCELLENT: Outstanding detection performance\n")
                improvement_suggestions = [
                    "Fine-tune for edge cases", "Consider ensemble methods"]
            elif map50 >= 0.7:
                f.write("🟢 VERY GOOD: Strong detection performance\n")
                improvement_suggestions = [
                    "Optimize inference speed", "Reduce false positives"]
            elif map50 >= 0.6:
                f.write(
                    "🟡 GOOD: Solid detection performance with room for improvement\n")
                improvement_suggestions = [
                    "Hyperparameter tuning", "Data augmentation", "Longer training"]
            elif map50 >= 0.4:
                f.write("🟠 MODERATE: Acceptable performance, needs optimization\n")
                improvement_suggestions = [
                    "Architecture changes", "Better data quality", "Advanced training techniques"]
            else:
                f.write("🔴 POOR: Significant improvement needed\n")
                improvement_suggestions = [
                    "Review model architecture", "Check data quality", "Increase training duration"]

            # Precision analysis
            if precision >= 0.85:
                f.write("✅ HIGH PRECISION: Excellent false positive control\n")
            elif precision >= 0.7:
                f.write("✅ GOOD PRECISION: Good false positive control\n")
            else:
                f.write("⚠️  LOW PRECISION: High false positive rate\n")

            # Recall analysis
            if recall >= 0.85:
                f.write("✅ HIGH RECALL: Excellent detection coverage\n")
            elif recall >= 0.7:
                f.write("✅ GOOD RECALL: Good detection coverage\n")
            else:
                f.write("⚠️  LOW RECALL: Missing many cattle instances\n")

            # Balance analysis
            precision_recall_diff = abs(precision - recall)
            if precision_recall_diff < 0.05:
                f.write("⚖️  BALANCED: Good precision-recall balance\n")
            elif precision > recall + 0.1:
                f.write(
                    "📈 PRECISION-BIASED: Conservative predictions, consider lowering threshold\n")
            else:
                f.write(
                    "📉 RECALL-BIASED: Aggressive predictions, consider raising threshold\n")

            f.write("\n")

            # Per-Class Detailed Metrics
            if per_class:
                f.write("📋 PER-CLASS DETAILED METRICS:\n")
                f.write("-" * 60 + "\n")
                for class_name, metrics in per_class.items():
                    if isinstance(metrics, dict):
                        f.write(f"\n{class_name.capitalize()}:\n")
                        f.write(
                            f"  Average Precision: {metrics.get('ap', 0):.4f} ({metrics.get('ap', 0)*100:.2f}%)\n")
                        f.write(
                            f"  Precision:         {metrics.get('precision', 0):.4f} ({metrics.get('precision', 0)*100:.2f}%)\n")
                        f.write(
                            f"  Recall:           {metrics.get('recall', 0):.4f} ({metrics.get('recall', 0)*100:.2f}%)\n")
                        f.write(
                            f"  F1-Score:         {metrics.get('f1', 0):.4f} ({metrics.get('f1', 0)*100:.2f}%)\n")
                f.write("\n")

            # IoU Threshold Analysis (Enhanced)
            iou_analysis = results.get('iou_analysis', {})
            if iou_analysis:
                f.write("🎯 IoU THRESHOLD ANALYSIS:\n")
                f.write("-" * 60 + "\n")
                f.write("Performance across different IoU thresholds:\n\n")
                f.write("IoU Thresh | Average Precision | Performance Grade\n")
                f.write("-----------+------------------+------------------\n")

                for iou_thresh, ap_value in iou_analysis.items():
                    if isinstance(ap_value, (int, float)):
                        grade = "🟢 Excellent" if ap_value >= 0.7 else "🟡 Good" if ap_value >= 0.5 else "🟠 Moderate" if ap_value >= 0.3 else "🔴 Poor"
                        f.write(
                            f"   {iou_thresh:.2f}     |     {ap_value:.4f}        | {grade}\n")
                f.write("\n")

            # Improvement Recommendations
            f.write("💡 RECOMMENDATIONS FOR IMPROVEMENT:\n")
            f.write("-" * 60 + "\n")
            for i, suggestion in enumerate(improvement_suggestions, 1):
                f.write(f"• {suggestion}\n")

            # Model-specific recommendations
            if map50 < 0.7:
                f.write("• Consider increasing training epochs or learning rate\n")
            if precision < 0.8:
                f.write(
                    "• Increase confidence threshold to reduce false positives\n")
            if recall < 0.7:
                f.write(
                    "• Lower confidence threshold or improve data augmentation\n")
            if map75 < map50 * 0.6:
                f.write(
                    "• Focus on improving localization accuracy (bounding box precision)\n")

            f.write("\n")

            # Training Progress (if epoch provided)
            if epoch is not None:
                f.write("📈 TRAINING PROGRESS:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Current epoch: {epoch}\n")
                f.write(
                    "• Check training_metrics.csv for detailed progress tracking\n")
                f.write("• Training curves available in training_curves.png\n\n")

            f.write("=" * 80 + "\n")
            f.write(
                "📊 End of Report - Check training_metrics.csv for consolidated tracking\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Enhanced evaluation report saved to {save_path}")

    @staticmethod
    def cleanup_old_metrics(metrics_dir: str, keep_final: bool = True, keep_csv: bool = True):
        """
        Clean up old individual epoch metric files, keeping only essential ones.

        Args:
            metrics_dir: Path to metrics directory
            keep_final: Whether to keep final evaluation files
            keep_csv: Whether to keep CSV tracking file
        """
        metrics_path = Path(metrics_dir)
        if not metrics_path.exists():
            logger.warning(f"Metrics directory not found: {metrics_path}")
            return

        files_removed = 0

        for file_path in metrics_path.iterdir():
            if file_path.is_file():
                filename = file_path.name

                # Keep CSV files
                if keep_csv and filename.endswith('.csv'):
                    continue

                # Keep final evaluation files
                if keep_final and ('final' in filename or 'training_curves' in filename):
                    continue

                # Remove individual epoch files
                if 'epoch_' in filename and (filename.endswith('.json') or filename.endswith('.txt')):
                    try:
                        file_path.unlink()
                        files_removed += 1
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")

        logger.info(
            f"Cleaned up {files_removed} old metric files from {metrics_path}")
        logger.info(f"Consolidated metrics available in training_metrics.csv")


def evaluate_model(model, dataloader, device, score_threshold=0.5, num_classes=2):
    """
    Evaluate a model on a dataset and return comprehensive metrics.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        score_threshold: Minimum confidence threshold
        num_classes: Number of classes

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    metrics = DetectionMetrics(
        score_threshold=score_threshold, num_classes=num_classes)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            # Get model predictions
            predictions = model(images)

            # Update metrics
            metrics.update(predictions, targets)

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Evaluated {batch_idx + 1} batches")

    # Compute and return results
    results = metrics.compute_metrics()
    return results, metrics


# Export commonly used functions
__all__ = [
    'DetectionMetrics',
    'MetricsTracker',
    'evaluate_model'
]

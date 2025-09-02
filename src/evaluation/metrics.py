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

logger = logging.getLogger(__name__)


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

    def save_metrics(self, results: Dict, save_path: str):
        """
        Save metrics results to a file.

        Args:
            results: Metrics results
            save_path: Path to save the metrics
        """
        import json

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

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Metrics saved to {save_path}")

        # Also save a human-readable text summary
        txt_path = save_path.replace('.json', '_summary.txt')
        self.save_text_summary(results, txt_path)

    def save_text_summary(self, results: Dict, save_path: str):
        """
        Save a human-readable text summary of metrics.

        Args:
            results: Metrics results
            save_path: Path to save the text summary
        """
        from datetime import datetime
        import platform

        summary = results.get('summary', {})
        per_class = results.get('per_class_metrics', {})

        with open(save_path, 'w') as f:
            # Header
            f.write("CATTLE DETECTION MODEL - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Timestamp and system info
            f.write(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System: {platform.system()} {platform.release()}\n")
            f.write(f"Python: {platform.python_version()}\n\n")

            # Dataset Statistics
            f.write("DATASET STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Images evaluated: {len(self.predictions)}\n")
            f.write(
                f"Total predictions: {summary.get('num_predictions', 0)}\n")
            f.write(
                f"Total ground truths: {summary.get('num_ground_truths', 0)}\n")
            f.write(
                f"Score threshold: {summary.get('score_threshold', self.score_threshold):.2f}\n\n")

            # Key Performance Metrics
            f.write("KEY PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"mAP@0.5       : {summary.get('mAP@0.5', 0):.4f} ({summary.get('mAP@0.5', 0)*100:.2f}%)\n")
            f.write(
                f"mAP@0.75      : {summary.get('mAP@0.75', 0):.4f} ({summary.get('mAP@0.75', 0)*100:.2f}%)\n")
            f.write(
                f"mAP@0.5:0.95  : {summary.get('mAP@0.5:0.95', 0):.4f} ({summary.get('mAP@0.5:0.95', 0)*100:.2f}%)\n")
            f.write(
                f"Precision@0.5 : {summary.get('precision@0.5', 0):.4f} ({summary.get('precision@0.5', 0)*100:.2f}%)\n")
            f.write(
                f"Recall@0.5    : {summary.get('recall@0.5', 0):.4f} ({summary.get('recall@0.5', 0)*100:.2f}%)\n")
            f.write(
                f"F1-Score@0.5  : {summary.get('f1@0.5', 0):.4f} ({summary.get('f1@0.5', 0)*100:.2f}%)\n\n")

            # Performance Interpretation
            f.write("PERFORMANCE INTERPRETATION:\n")
            f.write("-" * 30 + "\n")
            map50 = summary.get('mAP@0.5', 0)
            precision = summary.get('precision@0.5', 0)
            recall = summary.get('recall@0.5', 0)

            if map50 >= 0.8:
                f.write("🟢 EXCELLENT: Model shows excellent detection performance\n")
            elif map50 >= 0.6:
                f.write("🟡 GOOD: Model shows good detection performance\n")
            elif map50 >= 0.4:
                f.write("🟠 MODERATE: Model shows moderate detection performance\n")
            else:
                f.write("🔴 POOR: Model shows poor detection performance\n")

            if precision >= 0.8:
                f.write("✅ HIGH PRECISION: Low false positive rate\n")
            elif precision >= 0.6:
                f.write("⚡ MODERATE PRECISION: Acceptable false positive rate\n")
            else:
                f.write("⚠️  LOW PRECISION: High false positive rate\n")

            if recall >= 0.8:
                f.write("✅ HIGH RECALL: Successfully detects most cattle\n")
            elif recall >= 0.6:
                f.write("⚡ MODERATE RECALL: Misses some cattle instances\n")
            else:
                f.write("⚠️  LOW RECALL: Misses many cattle instances\n")

            f.write("\n")

            # Per-Class Metrics
            if per_class:
                f.write("PER-CLASS DETAILED METRICS:\n")
                f.write("-" * 30 + "\n")
                for class_id, class_metrics in per_class.items():
                    if 0.5 in class_metrics:
                        metrics = class_metrics[0.5]
                        class_name = f"Class {class_id}" if class_id != 1 else "Cattle"
                        f.write(f"\n{class_name}:\n")
                        f.write(
                            f"  Average Precision: {metrics['ap']:.4f} ({metrics['ap']*100:.2f}%)\n")
                        f.write(
                            f"  Precision:         {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                        f.write(
                            f"  Recall:           {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                        f.write(
                            f"  F1-Score:         {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)\n")

            # IoU Threshold Analysis
            f.write("\n\nIoU THRESHOLD ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write("Performance across different IoU thresholds:\n\n")

            if per_class and 1 in per_class:  # Assuming class 1 is cattle
                cattle_metrics = per_class[1]
                f.write("IoU Thresh | Average Precision\n")
                f.write("-----------+------------------\n")
                for iou_thresh in sorted(cattle_metrics.keys()):
                    ap = cattle_metrics[iou_thresh]['ap']
                    f.write(f"   {iou_thresh:.2f}     |     {ap:.4f}\n")

            # Recommendations
            f.write("\n\nRECOMMENDations FOR IMPROVEMENT:\n")
            f.write("-" * 30 + "\n")

            if map50 < 0.6:
                f.write("• Consider training for more epochs\n")
                f.write("• Try data augmentation techniques\n")
                f.write("• Check if dataset has sufficient quality annotations\n")

            if precision < 0.7:
                f.write("• Model has high false positive rate\n")
                f.write("• Consider increasing confidence threshold\n")
                f.write("• Review difficult negative samples\n")

            if recall < 0.7:
                f.write("• Model misses cattle instances\n")
                f.write("• Consider lowering confidence threshold\n")
                f.write("• Add more diverse training samples\n")

            if summary.get('mAP@0.75', 0) / max(map50, 0.001) < 0.7:
                f.write("• Localization accuracy can be improved\n")
                f.write("• Consider bbox regression loss tuning\n")
                f.write("• Check annotation quality for precise boundaries\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("End of Report\n")

        logger.info(f"Text summary saved to {save_path}")


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
    'evaluate_model'
]

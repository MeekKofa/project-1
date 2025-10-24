"""Utilities for detection evaluation: post-processing predictions and computing metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision.ops import box_iou, nms

# Import robust post-processing utilities
try:
    from src.utils.prediction_postprocessing import robust_postprocess_predictions
    HAS_ROBUST_POSTPROCESSING = True
except ImportError:
    HAS_ROBUST_POSTPROCESSING = False


def _compute_average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """Compute average precision given recall and precision curves."""
    if recall.numel() == 0:
        return 0.0

    device = recall.device
    mrec = torch.cat([torch.tensor([0.0], device=device),
                     recall, torch.tensor([1.0], device=device)])
    mpre = torch.cat([torch.tensor([0.0], device=device),
                     precision, torch.tensor([0.0], device=device)])

    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    indices = torch.nonzero(mrec[1:] != mrec[:-1]).flatten()
    ap = torch.sum((mrec[indices + 1] - mrec[indices])
                   * mpre[indices + 1]).item()
    return ap


def postprocess_batch_predictions(
    predictions: Any,
    image_shapes: List[Tuple[int, int]],
    conf_threshold: float = 0.5,
    nms_iou: float = 0.4,
    max_det: int = 300,
    use_robust_filtering: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    Convert raw model outputs into per-image detection dictionaries.

    Args:
        predictions: Model predictions (list of dicts or tuple)
        image_shapes: List of (height, width) for each image
        conf_threshold: Confidence threshold for filtering
        nms_iou: IoU threshold for NMS
        max_det: Maximum number of detections
        use_robust_filtering: Use robust post-processing to filter outside/invalid boxes

    Returns:
        List of prediction dicts with 'boxes', 'scores', 'labels'
    """
    # Case 1: model already returns list of dicts (e.g., Faster R-CNN)
    if isinstance(predictions, list) and predictions and isinstance(predictions[0], dict):
        processed = []
        for idx, pred in enumerate(predictions):
            boxes = pred.get('boxes', torch.empty(
                (0, 4), device=pred.get('scores', torch.tensor([])).device))
            scores = pred.get('scores', torch.ones(
                boxes.size(0), device=boxes.device))
            labels = pred.get('labels', torch.zeros(
                boxes.size(0), dtype=torch.int64, device=boxes.device))

            # DEBUG: Ensure scores are properly formatted
            # Faster R-CNN should already return scores, but verify they exist
            if scores.numel() > 0 and boxes.numel() > 0:
                assert scores.numel() == boxes.size(
                    0), f"Score/box mismatch: {scores.numel()} vs {boxes.size(0)}"

            result = {
                'boxes': boxes.detach(),
                'scores': scores.detach(),
                'labels': labels.detach(),
            }

            # Apply robust post-processing if available
            if use_robust_filtering and HAS_ROBUST_POSTPROCESSING and len(image_shapes) > idx:
                height, width = image_shapes[idx]
                result = robust_postprocess_predictions(
                    result,
                    image_width=width,
                    image_height=height,
                    conf_threshold=conf_threshold,
                    nms_iou_threshold=nms_iou,
                    max_detections=max_det,
                    min_box_size=10,
                    outside_margin=0.1
                )

            processed.append(result)
        return processed

    # Case 2: tuple output expected from YOLO-like models
    if not isinstance(predictions, tuple) or len(predictions) != 3:
        raise TypeError("Unsupported prediction format for post-processing")

    box_preds, cls_preds, obj_preds = predictions
    batch_size = box_preds.size(0)
    num_classes = cls_preds.size(2)

    results: List[Dict[str, torch.Tensor]] = []

    for idx in range(batch_size):
        boxes = box_preds[idx]
        cls_logits = cls_preds[idx]
        obj_logits = obj_preds[idx].squeeze(-1)

        obj_scores = torch.sigmoid(obj_logits)
        cls_scores = torch.sigmoid(cls_logits)
        combined_scores = obj_scores.unsqueeze(-1) * cls_scores

        height, width = image_shapes[idx]
        boxes_clamped = boxes.clone()
        boxes_clamped[:, [0, 2]] = boxes_clamped[:, [0, 2]].clamp_(0, width)
        boxes_clamped[:, [1, 3]] = boxes_clamped[:, [1, 3]].clamp_(0, height)

        boxes_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for class_idx in range(num_classes):
            cls_conf = combined_scores[:, class_idx]
            keep_mask = cls_conf > conf_threshold
            if keep_mask.sum() == 0:
                continue

            class_boxes = boxes_clamped[keep_mask]
            class_scores = cls_conf[keep_mask]
            kept_indices = nms(class_boxes, class_scores, nms_iou)
            if kept_indices.numel() == 0:
                continue

            kept_indices = kept_indices[:max_det]
            boxes_list.append(class_boxes[kept_indices])
            scores_list.append(class_scores[kept_indices])
            labels_list.append(torch.full(
                (kept_indices.numel(),), class_idx, dtype=torch.int64, device=class_boxes.device))

        if boxes_list:
            boxes_out = torch.cat(boxes_list, dim=0)
            scores_out = torch.cat(scores_list, dim=0)
            labels_out = torch.cat(labels_list, dim=0)

            order = torch.argsort(scores_out, descending=True)
            order = order[:max_det]
            results.append({
                'boxes': boxes_out[order].detach(),
                'scores': scores_out[order].detach(),
                'labels': labels_out[order].detach(),
            })
        else:
            device = boxes.device
            results.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=device),
            })

    return results


class DetectionMetricAccumulator:
    """Accumulate detection predictions/targets and compute evaluation metrics."""

    def __init__(self, num_classes: int, iou_threshold: float = 0.3) -> None:  # Temporarily reduced for early training
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.predictions: List[List[Tuple[int, float, torch.Tensor]]] = [
            [] for _ in range(num_classes)
        ]
        self.ground_truths: List[defaultdict[int, List[torch.Tensor]]] = [
            defaultdict(list) for _ in range(num_classes)
        ]
        self.sample_predictions: Dict[int, Dict[str, Any]] = {}
        self.sample_targets: Dict[int, Dict[str, Any]] = {}

    def update(self, predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]):
        """Add batch predictions and targets to the accumulator."""
        for pred, target in zip(predictions, targets):
            image_id_tensor = target.get('image_id')
            if isinstance(image_id_tensor, torch.Tensor):
                image_id = int(image_id_tensor.flatten()[0].item())
            elif image_id_tensor is not None:
                image_id = int(image_id_tensor)
            else:
                image_id = len(self.sample_predictions)

            pred_boxes = pred['boxes'].detach().cpu()
            pred_scores = pred['scores'].detach().cpu()
            pred_labels = pred['labels'].detach().cpu()

            gt_boxes = target['boxes'].detach().cpu()
            gt_labels = target['labels'].detach().cpu()

            # Store per-class data for metrics
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                self.predictions[label.item()].append(
                    (image_id, float(score.item()), box))

            for box, label in zip(gt_boxes, gt_labels):
                self.ground_truths[label.item()][image_id].append(box)

            # Persist per-sample predictions for later inspection
            self.sample_predictions[image_id] = {
                'boxes': pred_boxes.tolist(),
                'scores': pred_scores.tolist(),
                'labels': pred_labels.tolist(),
            }
            self.sample_targets[image_id] = {
                'boxes': gt_boxes.tolist(),
                'labels': gt_labels.tolist(),
            }

    def compute(self) -> Dict[str, Any]:
        """Compute global and per-class detection metrics."""
        eps = 1e-6
        aps: List[float] = []
        per_class_stats: List[Dict[str, Optional[float]]] = []
        total_tp = 0.0
        total_fp = 0.0
        total_gt = 0.0

        for class_idx in range(self.num_classes):
            class_preds = sorted(
                self.predictions[class_idx], key=lambda x: x[1], reverse=True)
            gt_dict = self.ground_truths[class_idx]
            gt_count = sum(len(boxes) for boxes in gt_dict.values())
            total_gt += gt_count

            if gt_count == 0:
                per_class_stats.append({
                    'class_id': class_idx,
                    'ap': None,
                    'precision': None,
                    'recall': None,
                    'tp': 0,
                    'fp': 0,
                    'gt': 0,
                })
                continue

            if not class_preds:
                aps.append(0.0)
                per_class_stats.append({
                    'class_id': class_idx,
                    'ap': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'tp': 0,
                    'fp': 0,
                    'gt': gt_count,
                })
                continue

            matched = {
                image_id: torch.zeros(len(boxes), dtype=torch.bool)
                for image_id, boxes in gt_dict.items()
            }

            tp = torch.zeros(len(class_preds))
            fp = torch.zeros(len(class_preds))

            for idx, (image_id, score, box) in enumerate(class_preds):
                if image_id not in gt_dict:
                    fp[idx] = 1
                    continue

                gt_boxes = torch.stack(gt_dict[image_id])
                ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_idx = torch.max(ious, dim=0)

                if best_iou >= self.iou_threshold and not matched[image_id][best_idx]:
                    tp[idx] = 1
                    matched[image_id][best_idx] = True
                else:
                    fp[idx] = 1

            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)

            recall = tp_cum / (gt_count + eps)
            precision = tp_cum / (tp_cum + fp_cum + eps)
            ap = _compute_average_precision(recall, precision)
            aps.append(ap)

            cls_tp = tp_cum[-1].item()
            cls_fp = fp_cum[-1].item()
            precision_cls = cls_tp / (cls_tp + cls_fp + eps)
            recall_cls = cls_tp / (gt_count + eps)

            total_tp += cls_tp
            total_fp += cls_fp

            per_class_stats.append({
                'class_id': class_idx,
                'ap': ap,
                'precision': precision_cls,
                'recall': recall_cls,
                'tp': cls_tp,
                'fp': cls_fp,
                'gt': gt_count,
            })

        map_50 = float(sum(aps) / len(aps)) if aps else 0.0
        precision_global = total_tp / \
            (total_tp + total_fp + eps) if (total_tp + total_fp) > 0 else 0.0
        recall_global = total_tp / (total_gt + eps) if total_gt > 0 else 0.0
        f1_global = 2 * precision_global * recall_global / \
            (precision_global + recall_global +
             eps) if (precision_global + recall_global) > 0 else 0.0

        return {
            'map_50': map_50,
            'precision_50': float(precision_global),
            'recall_50': float(recall_global),
            'f1_50': float(f1_global),
            'per_class': per_class_stats,
            'total_gt': int(total_gt),
            'total_predictions': int(sum(len(p) for p in self.predictions)),
        }

    def save_predictions(self, output_path: Path) -> None:
        """Persist accumulated predictions/targets to JSON for later analysis."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'predictions': self.sample_predictions,
            'targets': self.sample_targets,
        }
        with open(output_path, 'w') as fp:
            json.dump(payload, fp)

    def save_metrics(self, output_path: Path, metrics: Dict[str, Any], class_names: Optional[List[str]] = None) -> None:
        """Save metrics summary to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = metrics.copy()
        if class_names:
            for entry in summary.get('per_class', []):
                class_idx = entry['class_id']
                if class_idx < len(class_names):
                    entry['class_name'] = class_names[class_idx]
        with open(output_path, 'w') as fp:
            json.dump(summary, fp, indent=2)

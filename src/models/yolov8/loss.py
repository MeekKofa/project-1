"""
YOLOv8 Loss Function.

Clean, modular loss computation with proper separation of concerns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou
from typing import Dict, List, Tuple
import logging

from ...utils.box_utils import sanitize_boxes

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter for modulating loss
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: [N, C] predicted logits
            targets: [N, C] one-hot targets

        Returns:
            Scalar loss value
        """
        # Compute probabilities
        p = torch.sigmoid(logits)

        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Focal weight: (1 - p_t)^gamma
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final loss
        loss = alpha_t * focal_weight * bce_loss

        return loss.sum()


class YOLOv8Loss(nn.Module):
    """
    YOLOv8 Detection Loss.

    Combines:
    - Box regression loss (GIoU)
    - Classification loss (Focal Loss)
    - Objectness loss (BCE with positive/negative sampling)
    """

    def __init__(
        self,
        num_classes: int,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        obj_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        iou_thresh_start: float = 0.25,
        iou_thresh_end: float = 0.45,
        iou_warmup_epochs: int = 30
    ):
        """
        Initialize YOLOv8 loss.

        Args:
            num_classes: Number of object classes
            box_weight: Weight for box regression loss
            cls_weight: Weight for classification loss
            obj_weight: Weight for objectness loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            iou_thresh_start: Starting IoU threshold for positive matches
            iou_thresh_end: Final IoU threshold (after warmup)
            iou_warmup_epochs: Number of epochs for IoU threshold warmup
        """
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight

        # IoU threshold warmup
        self.iou_thresh_start = iou_thresh_start
        self.iou_thresh_end = iou_thresh_end
        self.iou_warmup_epochs = iou_warmup_epochs

        # Focal loss for classification
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Current epoch (for warmup)
        self.current_epoch = 0

    def get_current_iou_threshold(self) -> float:
        """Get current IoU threshold based on warmup schedule."""
        if self.current_epoch >= self.iou_warmup_epochs:
            return self.iou_thresh_end

        # Linear warmup
        progress = self.current_epoch / self.iou_warmup_epochs
        threshold = self.iou_thresh_start + progress * \
            (self.iou_thresh_end - self.iou_thresh_start)
        return threshold

    def match_predictions_to_targets(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match predictions to ground truth boxes using IoU.

        Args:
            pred_boxes: [N, 4] predicted boxes
            gt_boxes: [M, 4] ground truth boxes
            threshold: IoU threshold for positive matches

        Returns:
            Tuple of (pos_indices, matched_gt_indices, neg_mask)
        """
        if gt_boxes.numel() == 0:
            # No ground truth boxes
            return torch.empty(0, dtype=torch.long, device=pred_boxes.device), \
                torch.empty(0, dtype=torch.long, device=pred_boxes.device), \
                torch.ones(pred_boxes.size(0), dtype=torch.bool,
                           device=pred_boxes.device)

        if pred_boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=gt_boxes.device), \
                torch.empty(0, dtype=torch.long, device=gt_boxes.device), \
                torch.empty(0, dtype=torch.bool, device=gt_boxes.device)

        # Compute IoU between all predictions and ground truth
        ious = generalized_box_iou(pred_boxes, gt_boxes)  # [N, M]

        # For each prediction, find best matching ground truth
        best_ious, best_gt_idx = ious.max(dim=1)  # [N]

        # Positive matches: IoU > threshold
        pos_mask = best_ious > threshold
        pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
        matched_gt_idx: torch.Tensor

        if pos_indices.numel() == 0 and ious.numel() > 0:
            # Ensure each ground-truth gets at least one matched prediction
            fallback_matches = {}
            for gt_idx in range(gt_boxes.size(0)):
                column = ious[:, gt_idx]
                if column.numel() == 0:
                    continue
                best_pred_idx = torch.argmax(column)
                iou_value = column[best_pred_idx]
                key = int(best_pred_idx.item())
                current = fallback_matches.get(key)
                if current is None or iou_value > current[1]:
                    fallback_matches[key] = (gt_idx, iou_value)

            if fallback_matches:
                pos_indices = torch.tensor(
                    list(fallback_matches.keys()),
                    dtype=torch.long,
                    device=pred_boxes.device,
                )
                matched_gt_idx = torch.tensor(
                    [entry[0] for entry in fallback_matches.values()],
                    dtype=torch.long,
                    device=pred_boxes.device,
                )
            else:
                pos_indices = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
                matched_gt_idx = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
        else:
            matched_gt_idx = best_gt_idx[pos_indices]

        # Negative samples: IoU < 0.5 by default
        neg_mask = best_ious < 0.5
        if pos_indices.numel() > 0:
            neg_mask[pos_indices] = False

        return pos_indices, matched_gt_idx, neg_mask

    def compute_box_loss(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute box regression loss using GIoU.

        Args:
            pred_boxes: [N, 4] predicted boxes
            gt_boxes: [N, 4] ground truth boxes

        Returns:
            Scalar loss value
        """
        if pred_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        # Compute GIoU
        giou = generalized_box_iou(pred_boxes, gt_boxes).diag()  # [N]

        # Loss: 1 - GIoU
        loss = (1.0 - giou).sum()

        return loss

    def compute_classification_loss(
        self,
        pred_logits: torch.Tensor,
        gt_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss using Focal Loss.

        Args:
            pred_logits: [N, C] predicted class logits
            gt_labels: [N] ground truth class labels

        Returns:
            Scalar loss value
        """
        if pred_logits.numel() == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        # Create one-hot targets
        targets = torch.zeros_like(pred_logits)
        targets[range(len(gt_labels)), gt_labels] = 1.0

        # Compute focal loss
        loss = self.focal_loss(pred_logits, targets)

        return loss

    def compute_objectness_loss(
        self,
        obj_logits: torch.Tensor,
        pos_indices: torch.Tensor,
        neg_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute objectness loss using BCE.

        Args:
            obj_logits: [N] objectness logits
            pos_indices: Indices of positive samples
            neg_mask: Boolean mask of negative samples

        Returns:
            Scalar loss value
        """
        device = obj_logits.device

        # Create targets: 1 for positives, 0 for negatives
        obj_targets = torch.zeros_like(obj_logits)
        if pos_indices.numel() > 0:
            obj_targets[pos_indices] = 1.0

        # Compute loss for positives
        if pos_indices.numel() > 0:
            pos_loss = F.binary_cross_entropy_with_logits(
                obj_logits[pos_indices],
                obj_targets[pos_indices],
                reduction='sum'
            )
        else:
            pos_loss = torch.tensor(0.0, device=device)

        # Compute loss for negatives (with reduced weight)
        if neg_mask.sum() > 0:
            neg_loss = F.binary_cross_entropy_with_logits(
                obj_logits[neg_mask],
                obj_targets[neg_mask],
                reduction='sum'
            )
        else:
            neg_loss = torch.tensor(0.0, device=device)

        # Combine with reduced weight for negatives
        loss = pos_loss + 0.5 * neg_loss

        return loss

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            predictions: Tuple of (box_preds, cls_preds, obj_preds)
                - box_preds: [B, N, 4] predicted boxes in xyxy format
                - cls_preds: [B, N, C] predicted class logits
                - obj_preds: [B, N, 1] predicted objectness logits
            targets: List of B dicts, each containing:
                - 'boxes': [M_i, 4] ground truth boxes
                - 'labels': [M_i] ground truth class labels

        Returns:
            Dictionary of weighted loss components:
            - 'loss_box'
            - 'loss_cls'
            - 'loss_obj'
        """
        box_preds, cls_preds, obj_preds = predictions
        device = box_preds.device
        batch_size = box_preds.size(0)

        # Accumulate losses
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_obj_loss = 0.0
        total_positives = 0
        total_negatives = 0

        # Get current IoU threshold
        iou_threshold = self.get_current_iou_threshold()

        # Process each image in batch
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)

            if gt_boxes.numel() == 0:
                continue

            # Get predictions for this image
            pred_boxes_i = box_preds[i]  # [N, 4]
            pred_logits_i = cls_preds[i]  # [N, C]
            obj_logits_i = obj_preds[i].squeeze(-1)  # [N]

            # Sanitize boxes
            pred_boxes_i = sanitize_boxes(pred_boxes_i)
            gt_boxes = sanitize_boxes(gt_boxes)

            # Match predictions to ground truth
            pos_indices, matched_gt_idx, neg_mask = self.match_predictions_to_targets(
                pred_boxes_i, gt_boxes, iou_threshold
            )

            if pos_indices.numel() == 0:
                # No positive matches, only compute objectness loss
                obj_loss = self.compute_objectness_loss(
                    obj_logits_i, pos_indices, neg_mask)
                total_obj_loss += obj_loss
                total_negatives += neg_mask.sum().item()
                continue

            # Get matched predictions and targets
            matched_pred_boxes = pred_boxes_i[pos_indices]
            matched_gt_boxes = gt_boxes[matched_gt_idx]
            matched_pred_logits = pred_logits_i[pos_indices]
            matched_gt_labels = gt_labels[matched_gt_idx]

            # Compute losses
            box_loss = self.compute_box_loss(
                matched_pred_boxes, matched_gt_boxes)
            cls_loss = self.compute_classification_loss(
                matched_pred_logits, matched_gt_labels)
            obj_loss = self.compute_objectness_loss(
                obj_logits_i, pos_indices, neg_mask)

            # Accumulate
            total_box_loss += box_loss
            total_cls_loss += cls_loss
            total_obj_loss += obj_loss
            total_positives += pos_indices.numel()
            total_negatives += neg_mask.sum().item()

        # Handle edge case: no positives in entire batch
        if total_positives == 0:
            device = box_preds.device

            if total_negatives > 0:
                loss_obj = self.obj_weight * \
                    (total_obj_loss / max(total_negatives, 1))
            else:
                # Maintain gradient flow even with empty targets
                loss_obj = 0.01 * (box_preds.sum() * 0.0 + 1.0)

            zero = torch.tensor(0.0, device=device)
            return {
                'loss_box': zero,
                'loss_cls': zero,
                'loss_obj': loss_obj,
            }

        # Normalize and weight losses
        box_loss_normalized = total_box_loss / total_positives
        cls_loss_normalized = total_cls_loss / total_positives
        obj_loss_normalized = total_obj_loss / \
            max(total_positives + total_negatives, 1)

        # Weighted sum
        loss_box = self.box_weight * box_loss_normalized
        loss_cls = self.cls_weight * cls_loss_normalized
        loss_obj = self.obj_weight * obj_loss_normalized

        return {
            'loss_box': loss_box,
            'loss_cls': loss_cls,
            'loss_obj': loss_obj,
        }

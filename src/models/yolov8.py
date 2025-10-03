import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork, nms, box_iou, generalized_box_iou
import traceback
import logging
import torch.cuda.amp as amp
from collections import OrderedDict
import os
import random
from typing import Optional, Dict, Any
from PIL import Image
from torchvision.transforms import functional as TF


def _sanitize_boxes(boxes: torch.Tensor, max_size: float = None, eps: float = 1e-6) -> torch.Tensor:
    """Ensure boxes are valid for IoU computations."""
    if boxes.numel() == 0:
        return boxes

    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1e4, neginf=-1e4)

    x1 = torch.min(boxes[:, 0], boxes[:, 2])
    y1 = torch.min(boxes[:, 1], boxes[:, 3])
    x2 = torch.max(boxes[:, 0], boxes[:, 2])
    y2 = torch.max(boxes[:, 1], boxes[:, 3])

    if max_size is not None:
        x1 = x1.clamp(min=0.0, max=max_size)
        y1 = y1.clamp(min=0.0, max=max_size)
        x2 = x2.clamp(min=0.0, max=max_size)
        y2 = y2.clamp(min=0.0, max=max_size)

    x2 = torch.clamp(x2, min=x1 + eps)
    y2 = torch.clamp(y2, min=y1 + eps)

    return torch.stack([x1, y1, x2, y2], dim=-1)


class ResNet18_YOLOv8(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3, box_weight: float = 7.5, cls_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: use ResNet18 and keep up to layer4 (conv -> layer4)
        # Use pretrained=False for older PyTorch versions compatibility
        backbone = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-2])  # keep conv -> layer4

        # Feature map downsample factor (ResNet18 gives stride=32)
        self.out_channels = 512

        # Detection heads (single-scale)
        self.box_head = nn.Sequential(
            nn.Conv2d(self.out_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=1)  # bbox regression: (x, y, w, h)
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.out_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1)  # classification
        )

        # Objectness head - predicts object vs background
        self.obj_head = nn.Sequential(
            nn.Conv2d(self.out_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)  # objectness score
        )

        # loss criterion instance with configured weights
        self.criterion = YOLOLoss(
            num_classes=num_classes, box_weight=box_weight, cls_weight=cls_weight)

    def forward(self, x, targets=None):
        # Feature extraction
        feats = self.backbone(x)  # [B, 512, H/32, W/32]

        # Predictions
        box_preds = self.box_head(feats)   # [B, 4, H/32, W/32]
        cls_preds = self.cls_head(feats)   # [B, num_classes, H/32, W/32]
        obj_preds = self.obj_head(feats)   # [B, 1, H/32, W/32]

        # Flatten
        B, _, H, W = box_preds.shape
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(
            B, -1, 4)          # [B, N, 4]
        cls_preds = cls_preds.permute(0, 2, 3, 1).reshape(
            B, -1, self.num_classes)  # [B, N, C]
        obj_preds = obj_preds.permute(0, 2, 3, 1).reshape(
            B, -1, 1)          # [B, N, 1]

        # ---- Decode raw outputs (tx,ty,tw,th) -> xyxy in image pixels using grid+stride ----
        # assume feature map stride relative to input (ResNet18 final stride)
        stride = 32
        # create grid centers (0.5 offset), shape [H, W]
        grid_x = torch.arange(
            W, device=box_preds.device).repeat(H, 1)  # [H, W]
        grid_y = torch.arange(H, device=box_preds.device).unsqueeze(
            1).repeat(1, W)  # [H, W]
        cx_grid = (grid_x + 0.5) * stride  # [H, W]
        cy_grid = (grid_y + 0.5) * stride  # [H, W]
        # flatten and expand to batch: [1, H*W, 1] -> [B, H*W, 1]
        cx = cx_grid.reshape(1, -1, 1).expand(B, -1, 1)
        cy = cy_grid.reshape(1, -1, 1).expand(B, -1, 1)

        # tx,ty,tw,th from network
        tx, ty, tw, th = box_preds.split(1, dim=-1)  # each [B, N, 1]

        # constrain raw predictions to prevent numerical instabilities
        tx = tx.clamp(min=-10.0, max=10.0)
        ty = ty.clamp(min=-10.0, max=10.0)
        tw = tw.clamp(min=-10.0, max=10.0)
        th = th.clamp(min=-10.0, max=10.0)

        # center decode: sigmoid + grid offset, scaled by stride
        bx = torch.sigmoid(tx) * stride + cx
        by = torch.sigmoid(ty) * stride + cy
        # size decode: clamp then exponent to prevent box explosions
        # Clamp to [-4, 4] means box size range: [0.018 * stride, 54.6 * stride]
        tw_clamped = tw.clamp(min=-4.0, max=4.0)
        th_clamped = th.clamp(min=-4.0, max=4.0)
        bw = torch.exp(tw_clamped) * stride
        bh = torch.exp(th_clamped) * stride

        # xyxy ordering
        x1 = bx - bw / 2.0
        y1 = by - bh / 2.0
        x2 = bx + bw / 2.0
        y2 = by + bh / 2.0
        box_preds = torch.cat([x1, y1, x2, y2], dim=-1)  # [B, N, 4]

        # ensure valid ordering to avoid downstream assertion errors
        x_min = torch.minimum(box_preds[..., 0:1], box_preds[..., 2:3])
        y_min = torch.minimum(box_preds[..., 1:2], box_preds[..., 3:4])
        x_max = torch.maximum(box_preds[..., 0:1], box_preds[..., 2:3])
        y_max = torch.maximum(box_preds[..., 1:2], box_preds[..., 3:4])
        box_preds = torch.cat([x_min, y_min, x_max, y_max], dim=-1)

        if self.training or targets is not None:
            # training path returns raw preds for external loss
            return box_preds, cls_preds, obj_preds
        else:
            # Inference: combine objectness with class scores for final confidence
            obj_scores = torch.sigmoid(obj_preds.squeeze(-1))  # [B, N]
            cls_scores = F.softmax(cls_preds, dim=-1)  # [B, N, C]
            max_cls_scores, labels = cls_scores.max(dim=-1)  # [B, N]

            # Final confidence = objectness * class_score
            final_scores = obj_scores * max_cls_scores  # [B, N]

            detections = []
            for b in range(B):
                detections.append({
                    "boxes": box_preds[b],   # [N, 4] (x1,y1,x2,y2)
                    "scores": final_scores[b],  # [N] - combined confidence
                    "labels": labels[b]      # [N]
                })
            return detections

    def compute_loss(self, box_preds, cls_preds, obj_preds, targets):
        # delegate to criterion
        return self.criterion(box_preds, cls_preds, obj_preds, targets)

# Add YOLOLoss helper (placed before Dataset)


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, box_weight=5.0, cls_weight=1.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.alpha = alpha
        self.gamma = gamma  # Standard focal loss gamma

    def focal_loss(self, logits, targets):
        """
        Focal loss for multi-class binary classification.
        logits: [N, C]
        targets: [N, C] one-hot
        """
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss
        return loss.sum()

    def forward(self, box_preds, cls_preds, obj_preds, targets):
        """
        box_preds: [B, N, 4] (x1,y1,x2,y2)
        cls_preds: [B, N, C] (logits)
        obj_preds: [B, N, 1] (objectness logits)
        targets: dict of lists {"boxes": [B], "labels": [B]}
        """
        device = box_preds.device
        total_box_loss, total_cls_loss, total_obj_loss, num_pos, num_neg = 0.0, 0.0, 0.0, 0, 0

        for b in range(len(targets["boxes"])):
            gt_boxes = targets["boxes"][b].to(device)   # [n_i, 4]
            gt_labels = targets["labels"][b].to(device)  # [n_i]
            num_gts = gt_boxes.size(0)
            if num_gts == 0:
                continue

            preds_b = box_preds[b]      # [N, 4]
            logits_b = cls_preds[b]     # [N, C]
            obj_b = obj_preds[b].squeeze(-1)  # [N]

            # Sanitize boxes before IoU computation to avoid assertion errors
            preds_b = _sanitize_boxes(preds_b)
            gt_boxes = _sanitize_boxes(gt_boxes)

            # ---- IoU matching ----
            ious = generalized_box_iou(preds_b, gt_boxes)   # [N, n_i]
            best_gt_idx = ious.argmax(dim=1)                # best GT per pred
            best_ious = ious.max(dim=1).values

            # ---- dynamic IoU threshold warmup ----
            # Balanced threshold: 0.25 → 0.45 over 30 epochs
            # Not too low (avoids bad matches) not too high (allows learning)
            current_epoch = getattr(self, "current_epoch", 0)
            thr = 0.25 + min(current_epoch / 30.0, 1.0) * \
                (0.45 - 0.25)  # ramp 0.25 → 0.45 over 30 epochs
            pos_mask = best_ious > thr

            if pos_mask.sum() == 0:
                # fallback: take top matches to provide signal at early epochs
                valid_ious = best_ious.clamp(min=0)
                topk = min(5, valid_ious.numel())
                if topk == 0:
                    continue
                top_indices = torch.topk(valid_ious, k=topk).indices
                # filter out zero IoU selections
                positive_indices = top_indices[valid_ious[top_indices] > 0]
                if positive_indices.numel() == 0:
                    continue
                pos_indices = positive_indices
            else:
                pos_indices = torch.nonzero(
                    pos_mask, as_tuple=False).squeeze(1)

            matched_gt = gt_boxes[best_gt_idx[pos_indices]]
            matched_labels = gt_labels[best_gt_idx[pos_indices]]
            pred_boxes = preds_b[pos_indices]
            pred_logits = logits_b[pos_indices]

            pred_boxes = _sanitize_boxes(pred_boxes)
            matched_gt = _sanitize_boxes(matched_gt)

            # ---- Box regression loss (GIoU) ----
            giou = generalized_box_iou(
                pred_boxes, matched_gt).diag()  # [n_pos]
            box_loss = (1.0 - giou).sum()

            # ---- Classification loss (Focal Loss) ----
            target_cls = torch.zeros_like(pred_logits, device=device)
            target_cls[range(len(matched_labels)), matched_labels] = 1.0
            cls_loss = self.focal_loss(pred_logits, target_cls)

            # ---- Objectness loss (Binary CE with focal weighting) ----
            # Positive samples: matched predictions should have high objectness
            obj_target = torch.zeros_like(obj_b, device=device)
            obj_target[pos_indices] = 1.0

            # Negative samples: unmatched predictions with low IoU
            neg_mask = best_ious < 0.5  # IoU < 0.5 are clear negatives

            # Focal BCE for objectness with safety checks
            if len(pos_indices) > 0:
                obj_loss_pos = F.binary_cross_entropy_with_logits(
                    obj_b[pos_indices], obj_target[pos_indices], reduction='sum'
                )
            else:
                obj_loss_pos = torch.tensor(0.0, device=device)

            if neg_mask.sum() > 0:
                obj_loss_neg = F.binary_cross_entropy_with_logits(
                    obj_b[neg_mask], obj_target[neg_mask], reduction='sum'
                )
            else:
                obj_loss_neg = torch.tensor(0.0, device=device)

            obj_loss = obj_loss_pos + 0.5 * obj_loss_neg  # Reduce negative weight

            total_box_loss += box_loss
            total_cls_loss += cls_loss
            total_obj_loss += obj_loss
            num_pos += len(matched_gt)
            num_neg += neg_mask.sum().item()

        # Always return a valid loss, even with no positives (use objectness loss only)
        if num_pos == 0:
            if num_neg > 0 and total_obj_loss > 0:
                # Return small objectness-only loss to keep training stable
                return 0.1 * total_obj_loss / num_neg
            else:
                # Return tiny loss to avoid None - use box_preds to maintain grad graph
                return 0.01 * box_preds.sum() * 0.0 + 0.01

        # Normalize by number of positive samples, add objectness term
        loss = (
            self.box_weight * total_box_loss / num_pos +
            self.cls_weight * total_cls_loss / num_pos +
            # Objectness loss weight = 1.0
            1.0 * total_obj_loss / max(num_pos + num_neg, 1)
        )
        return loss


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transform=None,
        target_size=(384, 384),
        augment: bool = False,
        augmentation_params: Optional[Dict[str, Any]] = None
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size
        self.images = [f for f in os.listdir(
            image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.augment = augment

        default_aug = {
            "flip_prob": 0.5,
            "color_prob": 0.8,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "gamma_range": 0.2
        }
        if augmentation_params:
            default_aug.update({k: v for k, v in augmentation_params.items()
                                if k in default_aug})
        self.aug_params = default_aug

    def __len__(self):
        return len(self.images)

    def _letterbox_resize(self, img: Image.Image, target_size: tuple, fill_value: int = 114):
        """
        Resize image with aspect ratio preservation using letterbox (padding).
        Returns: resized_img, scale, (pad_left, pad_top)
        """
        orig_w, orig_h = img.size
        target_w, target_h = target_size

        # Calculate scaling factor to fit image inside target
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize with aspect ratio preserved
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Create padded canvas
        padded_img = Image.new(
            'RGB', target_size, (fill_value, fill_value, fill_value))

        # Center the resized image
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        padded_img.paste(img_resized, (pad_left, pad_top))

        return padded_img, scale, (pad_left, pad_top)

    def _apply_augmentations(self, img: Image.Image, boxes: torch.Tensor, orig_w: int, orig_h: int):
        """Apply simple geometric and photometric augmentations."""
        if not self.augment:
            return img, boxes

        # Horizontal flip
        if random.random() < self.aug_params.get("flip_prob", 0.5):
            img = TF.hflip(img)
            if boxes.numel() > 0:
                flipped = boxes.clone()
                flipped[:, 0] = orig_w - boxes[:, 2]
                flipped[:, 2] = orig_w - boxes[:, 0]
                boxes = flipped

        if random.random() < self.aug_params.get("color_prob", 0.8):
            hsv_h = float(self.aug_params.get("hsv_h", 0.0))
            hsv_s = float(self.aug_params.get("hsv_s", 0.0))
            hsv_v = float(self.aug_params.get("hsv_v", 0.0))
            gamma_range = float(self.aug_params.get("gamma_range", 0.0))

            if hsv_v > 0:
                factor = 1.0 + random.uniform(-hsv_v, hsv_v)
                img = TF.adjust_brightness(img, max(0.1, factor))
            if hsv_s > 0:
                factor = 1.0 + \
                    random.uniform(-min(hsv_s, 0.9), min(hsv_s, 0.9))
                img = TF.adjust_saturation(img, max(0.1, factor))
            if hsv_h > 0:
                hue_shift = random.uniform(-hsv_h, hsv_h)
                img = TF.adjust_hue(img, max(-0.5, min(0.5, hue_shift)))
            if gamma_range > 0:
                gamma = 1.0 + random.uniform(-gamma_range, gamma_range)
                img = TF.adjust_gamma(img, max(0.1, gamma))

        return img, boxes

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        lbl_path = os.path.join(
            self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        # ---- Load image ----
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        tgt_w, tgt_h = self.target_size

        # ---- Load labels ----
        boxes, labels = [], []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = map(float, parts)
                    cls = int(cls)

                    # YOLO txt gives normalized cx,cy,w,h ∈ [0,1]
                    cx, cy, bw, bh = x * orig_w, y * orig_h, w * orig_w, h * orig_h
                    x1, y1 = cx - bw / 2, cy - bh / 2
                    x2, y2 = cx + bw / 2, cy + bh / 2

                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(
            (0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(
            (0,), dtype=torch.int64)

        # augment before resizing to preserve aspect-based calculations
        img, boxes = self._apply_augmentations(img, boxes, orig_w, orig_h)

        # ---- Letterbox resize (preserves aspect ratio with padding) ----
        img_resized, scale, (pad_left, pad_top) = self._letterbox_resize(
            img, (tgt_w, tgt_h))

        if self.transform:
            img_resized = self.transform(img_resized)

        # ---- Transform boxes to match letterbox coordinates ----
        if boxes.numel() > 0:
            # Scale boxes
            boxes = boxes * scale
            # Apply padding offset
            boxes[:, [0, 2]] += pad_left
            boxes[:, [1, 3]] += pad_top
            # Clamp to image bounds
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, tgt_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, tgt_h)

        target = {"boxes": boxes, "labels": labels}
        return img_resized, target

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
from PIL import Image


class ResNet18_YOLOv8(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
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

        # loss criterion instance
        self.criterion = YOLOLoss(num_classes=num_classes)

    def forward(self, x, targets=None):
        # Feature extraction
        feats = self.backbone(x)  # [B, 512, H/32, W/32]

        # Predictions
        box_preds = self.box_head(feats)   # [B, 4, H/32, W/32]
        cls_preds = self.cls_head(feats)   # [B, num_classes, H/32, W/32]

        # Flatten
        B, _, H, W = box_preds.shape
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(
            B, -1, 4)          # [B, N, 4]
        cls_preds = cls_preds.permute(0, 2, 3, 1).reshape(
            B, -1, self.num_classes)  # [B, N, C]

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
        # center decode: sigmoid + grid offset, scaled by stride
        bx = torch.sigmoid(tx) * stride + cx
        by = torch.sigmoid(ty) * stride + cy
        # size decode: exponent and scale by stride
        bw = torch.exp(tw) * stride
        bh = torch.exp(th) * stride

        # xyxy
        x1 = bx - bw / 2.0
        y1 = by - bh / 2.0
        x2 = bx + bw / 2.0
        y2 = by + bh / 2.0
        box_preds = torch.cat([x1, y1, x2, y2], dim=-1)  # [B, N, 4]

        if self.training or targets is not None:
            # training path returns raw preds for external loss
            return box_preds, cls_preds
        else:
            # Inference: return detection-style list of dicts (for evaluate)
            scores = F.softmax(cls_preds, dim=-1)  # [B, N, C]
            max_scores, labels = scores.max(dim=-1)  # [B, N]

            detections = []
            for b in range(B):
                detections.append({
                    "boxes": box_preds[b],   # [N, 4] (x1,y1,x2,y2)
                    "scores": max_scores[b],  # [N]
                    "labels": labels[b]      # [N]
                })
            return detections

    def compute_loss(self, box_preds, cls_preds, targets):
        # delegate to criterion
        return self.criterion(box_preds, cls_preds, targets)

# Add YOLOLoss helper (placed before Dataset)


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, box_weight=5.0, cls_weight=1.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.alpha = alpha
        self.gamma = gamma

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

    def forward(self, box_preds, cls_preds, targets):
        """
        box_preds: [B, N, 4] (x1,y1,x2,y2)
        cls_preds: [B, N, C] (logits)
        targets: dict of lists {"boxes": [B], "labels": [B]}
        """
        device = box_preds.device
        total_box_loss, total_cls_loss, num_samples = 0.0, 0.0, 0

        for b in range(len(targets["boxes"])):
            gt_boxes = targets["boxes"][b].to(device)   # [n_i, 4]
            gt_labels = targets["labels"][b].to(device)  # [n_i]
            num_gts = gt_boxes.size(0)
            if num_gts == 0:
                continue

            preds_b = box_preds[b]      # [N, 4]
            logits_b = cls_preds[b]     # [N, C]

            # ---- IoU matching ----
            ious = generalized_box_iou(preds_b, gt_boxes)   # [N, n_i]
            best_gt_idx = ious.argmax(dim=1)                # best GT per pred
            best_ious = ious.max(dim=1).values

            # ---- dynamic IoU threshold warmup ----
            current_epoch = getattr(self, "current_epoch", 0)
            thr = 0.3 + min(current_epoch / 10.0, 1.0) * \
                (0.5 - 0.3)  # ramp 0.3 → 0.5 over 10 epochs
            pos_mask = best_ious > thr

            if pos_mask.sum() == 0:
                continue

            matched_gt = gt_boxes[best_gt_idx[pos_mask]]
            matched_labels = gt_labels[best_gt_idx[pos_mask]]
            pred_boxes = preds_b[pos_mask]
            pred_logits = logits_b[pos_mask]

            # ---- Box regression loss (GIoU) ----
            giou = generalized_box_iou(
                pred_boxes, matched_gt).diag()  # [n_pos]
            box_loss = (1.0 - giou).sum()

            # ---- Classification loss (Focal Loss) ----
            target_cls = torch.zeros_like(pred_logits, device=device)
            target_cls[range(len(matched_labels)), matched_labels] = 1.0
            cls_loss = self.focal_loss(pred_logits, target_cls)

            total_box_loss += box_loss
            total_cls_loss += cls_loss
            num_samples += len(matched_gt)

        if num_samples == 0:
            return None

        loss = (self.box_weight * total_box_loss +
                self.cls_weight * total_cls_loss) / num_samples
        return loss


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_size=(384, 384)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size
        self.images = [f for f in os.listdir(
            image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self):
        return len(self.images)

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

        # ---- Resize image to target ----
        img = img.resize((tgt_w, tgt_h))
        if self.transform:
            img = self.transform(img)

        # ---- Rescale boxes to resized image ----
        if boxes.numel() > 0:
            scale_x, scale_y = tgt_w / orig_w, tgt_h / orig_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        target = {"boxes": boxes, "labels": labels}
        return img, target

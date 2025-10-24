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


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.box_head = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        box = self.box_head(x)
        cls = self.cls_head(x)
        return box, cls

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

        # Multi-Scale Heads (FPN/YOLO-style)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512],  # from ResNet layers 2,3,4
            out_channels=256
        )
        self.detect_heads = nn.ModuleList([
            DetectionHead(256, num_classes),  # for P3
            DetectionHead(256, num_classes),  # for P4
            DetectionHead(256, num_classes)   # for P5
        ])

        # loss criterion instance
        self.criterion = YOLOLoss(num_classes=num_classes)

    def forward(self, x, targets=None):
        # Correctly extract features for FPN
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # bn1
        x = self.backbone[2](x)  # relu
        x = self.backbone[3](x)  # maxpool
        f1 = self.backbone[4](x)  # layer1, 64 channels
        f2 = self.backbone[5](f1) # layer2, 128 channels
        f3 = self.backbone[6](f2) # layer3, 256 channels
        f4 = self.backbone[7](f3) # layer4, 512 channels

        # FPN forward
        fpn_feats = self.fpn(OrderedDict([
            ('0', f2),  # from layer2
            ('1', f3),  # from layer3
            ('2', f4)   # from layer4
        ]))

        # Predictions from each FPN level
        all_boxes, all_classes = [], []
        feat_shapes = []
        strides = [8, 16, 32]  # Strides of layer2, layer3, layer4 outputs
        for i, (feat, head, stride) in enumerate(zip(fpn_feats.values(), self.detect_heads, strides)):
            box_pred_raw, cls_pred_raw = head(feat)

            B, _, H, W = box_pred_raw.shape
            feat_shapes.append((H, W))
            
            # Reshape for decoding
            box_pred = box_pred_raw.permute(0, 2, 3, 1).reshape(B, -1, 4)
            cls_pred = cls_pred_raw.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)

            # Create grid for this level
            grid_x = torch.arange(W, device=box_pred.device).repeat(H, 1)
            grid_y = torch.arange(H, device=box_pred.device).unsqueeze(1).repeat(1, W)
            cx = (grid_x + 0.5) * stride
            cy = (grid_y + 0.5) * stride
            cx = cx.reshape(1, -1, 1).expand(B, -1, 1)
            cy = cy.reshape(1, -1, 1).expand(B, -1, 1)

            # Decode box predictions
            tx, ty, tw, th = box_pred.split(1, dim=-1)
            bx = torch.sigmoid(tx) * stride + cx
            by = torch.sigmoid(ty) * stride + cy
            bw = torch.exp(tw) * stride
            bh = torch.exp(th) * stride
            
            # Convert to xyxy
            box_pred_decoded = torch.cat([bx - bw/2, by - bh/2, bx + bw/2, by + bh/2], dim=-1)

            all_boxes.append(box_pred_decoded)
            all_classes.append(cls_pred)

        # Concatenate all levels
        box_preds = torch.cat(all_boxes, dim=1)
        cls_preds = torch.cat(all_classes, dim=1)

        if self.training or targets is not None:
            # training path returns raw preds for external loss
            return box_preds, cls_preds, feat_shapes, strides
        else:
            # Inference: return detection-style list of dicts (for evaluate)
            scores = cls_preds.sigmoid()  # (B, N, C)
            B, N, C = scores.shape
            boxes = box_preds.unsqueeze(2).expand(B, N, C, 4)  # match classes
            scores = scores.reshape(B, -1)                     # (B, N*C)
            boxes = boxes.reshape(B, -1, 4)
            labels = torch.arange(C, device=cls_preds.device).repeat(N)
            labels = labels.unsqueeze(0).expand(B, -1)         # (B, N*C)

            detections = torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).float()], dim=-1)
            return detections

    def compute_loss(self, box_preds, cls_preds, targets, feat_shapes, strides):
        # delegate to criterion
        return self.criterion(box_preds, cls_preds, targets, feat_shapes, strides)

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
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss
        return loss.sum()

    def forward(self, box_preds, cls_preds, targets, feat_shapes, strides):
        """
        box_preds: [B, N, 4]   (decoded boxes xyxy)
        cls_preds: [B, N, C]   (logits)
        targets:   dict with {"boxes": [B], "labels": [B]}
        feat_shapes: list of (H, W) for each FPN level
        strides: list of strides for each FPN level
        """
        device = box_preds.device
        B, N, C = cls_preds.shape

        total_box_loss, total_cls_loss, num_pos = 0.0, 0.0, 0

        for b in range(B):
            gt_boxes = targets["boxes"][b].to(device)   # [n_i, 4]
            gt_labels = targets["labels"][b].to(device) # [n_i]
            if gt_boxes.numel() == 0:
                continue

            preds_b = box_preds[b]   # [N, 4]
            logits_b = cls_preds[b]  # [N, C]

            offset = 0
            for (H, W), stride in zip(feat_shapes, strides):
                num_preds_lvl = H * W
                preds_lvl = preds_b[offset : offset + num_preds_lvl]
                logits_lvl = logits_b[offset : offset + num_preds_lvl]

                for gt, label in zip(gt_boxes, gt_labels):
                    # get GT center
                    cx = (gt[0] + gt[2]) / 2.0
                    cy = (gt[1] + gt[3]) / 2.0

                    # find grid cell (integer index in feature map)
                    gi = int(cx / stride)
                    gj = int(cy / stride)
                    if gi >= W or gj >= H:
                        continue
                    idx = gj * W + gi  # flatten to 1D index for this level

                    # ---- box regression loss ----
                    pred_box = preds_lvl[idx].unsqueeze(0)  # [1, 4]
                    giou = generalized_box_iou(pred_box, gt.unsqueeze(0)).diag()
                    box_loss = (1.0 - giou).sum()

                    # ---- classification loss ----
                    pred_logits = logits_lvl[idx].unsqueeze(0)  # [1, C]
                    target_cls = torch.zeros_like(pred_logits, device=device)
                    target_cls[0, label] = 1.0
                    cls_loss = self.focal_loss(pred_logits, target_cls)

                    total_box_loss += box_loss
                    total_cls_loss += cls_loss
                    num_pos += 1
                
                offset += num_preds_lvl

        if num_pos == 0:
            return None

        loss = (self.box_weight * total_box_loss +
                self.cls_weight * total_cls_loss) / num_pos
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
        if boxes.numel() > 0:
            scale_x, scale_y = tgt_w / orig_w, tgt_h / orig_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        target = {"boxes": boxes, "labels": labels}
        return img, target

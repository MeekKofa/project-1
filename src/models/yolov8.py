import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork, nms
import traceback
import logging
import torch.cuda.amp as amp
from collections import OrderedDict

class ResNet18_YOLOv8(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        # Essential attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = torch.cuda.is_available()
        self.debug = True
        self.grids = {}
        self.grid_sizes = [(48, 48), (24, 24), (12, 12)]  # For 384x384 input
        self.input_size = (384, 384)  # Reduced input size
        self.num_classes = num_classes
        
        # Initialize backbone with correct output channels
        backbone = models.resnet18(weights='DEFAULT')
        self.layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,  # 64 channels
            backbone.layer2   # 128 channels
        )
        self.layer2 = backbone.layer3  # 256 channels
        self.layer3 = backbone.layer4  # 512 channels

        # Add missing backbone_layers attribute
        self.backbone_layers = [self.layer1, self.layer2, self.layer3]
        self.num_anchors = 1  # Simplified anchor scheme
        
        # Fix channel sizes and FPN
        self.channels = {
            'layer3': 512,
            'layer2': 256,
            'layer1': 128
        }
        
        # Simpler FPN design
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for in_ch in [512, 256, 128]
        ])
        
        # Remove redundant FPN
        self.fpn = None
        self.lateral_convs = None

        # Update prediction heads for 256 channels
        self.box_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 4, 1)
            ) for _ in range(3)
        ])
        
        self.cls_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, 1)
            ) for _ in range(3)
        ])

        # Remove det_heads as we're using separate box and cls heads
        self.det_heads = None

        # Initialize weights
        self._initialize_weights()
        
        # Loss weights
        self.box_weight = 5.0
        self.cls_weight = 1.0

        self.register_buffer('anchor_grid', torch.zeros(1))  # Add anchor grid buffer

    def _initialize_weights(self):
        """Initialize weights for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_grid(self, h: int, w: int):
        """Initialize detection grids"""
        if f"{h}_{w}" not in self.grids:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing='ij'
            )
            grid = torch.stack((grid_x, grid_y), dim=2).float()
            self.grids[f"{h}_{w}"] = grid.view(1, -1, 2)

    def _get_grid(self, h: int, w: int, device: torch.device):
        """Get grid for given size"""
        key = f"{h}_{w}"
        if key not in self.grids:
            self._init_grid(h, w)
        return self.grids[key].to(device)

    def post_process(self, detections, conf_thres=0.25, iou_thres=0.45):
        """Post process detections with NMS"""
        processed_dets = []
        
        for det in detections:
            # Reshape detection maps [B, anchors*(5+classes), H, W] -> [B, H, W, anchors, 5+classes]
            B, C, H, W = det.shape
            det = det.permute(0, 2, 3, 1).reshape(B, H, W, self.num_anchors, -1)
            
            # Extract boxes, scores, classes
            box_xy = torch.sigmoid(det[..., 0:2])  # center x, y
            box_wh = torch.exp(det[..., 2:4])  # width, height
            conf = torch.sigmoid(det[..., 4])  # confidence
            cls_prob = torch.sigmoid(det[..., 5:])  # class probabilities
            
            # Convert to x1y1x2y2 format
            x1y1 = box_xy - box_wh / 2
            x2y2 = box_xy + box_wh / 2
            boxes = torch.cat([x1y1, x2y2], dim=-1)
            
            # Filter by confidence
            conf_mask = conf > conf_thres
            boxes = boxes[conf_mask]
            scores = conf[conf_mask] * cls_prob[conf_mask].max(dim=-1)[0]
            
            # Apply NMS
            keep = nms(boxes, scores, iou_thres)
            processed_dets.append({
                'boxes': boxes[keep],
                'scores': scores[keep],
                'classes': cls_prob[conf_mask][keep].argmax(dim=-1)
            })
            
        return processed_dets

    def inference(self, x):
        """Fixed inference method"""
        try:
            # Extract backbone features
            features = []
            for layer in self.backbone_layers:
                x = layer(x)
                features.append(x)
            
            # Process features through FPN
            fpn_outs = []
            for i, (feat, fpn_conv) in enumerate(zip(features, self.fpn_convs)):
                feat = fpn_conv(feat)
                if i > 0:
                    feat = feat + F.interpolate(fpn_outs[-1], size=feat.shape[-2:], mode='nearest')
                fpn_outs.append(feat)
            
            # Generate predictions
            box_preds_list = []
            cls_preds_list = []
            
            for feat, box_head, cls_head in zip(fpn_outs, self.box_head, self.cls_head):
                box_pred = box_head(feat)
                cls_pred = cls_head(feat)
                box_preds_list.append(box_pred)
                cls_preds_list.append(cls_pred)
            
            # Post-process predictions
            return self.post_process([torch.cat([b, c], dim=1) for b, c in zip(box_preds_list, cls_preds_list)])
            
        except Exception as e:
            print(f"Inference error: {str(e)}")
            print(traceback.format_exc())
            return None

    def _validate_shapes(self, box_preds, cls_preds, target_boxes):
        """Validate tensor shapes before loss computation"""
        print(f"[DEBUG] Shape validation:")
        print(f"  box_preds: {box_preds.shape}")
        print(f"  cls_preds: {cls_preds.shape}")
        print(f"  target_boxes: {target_boxes.shape}")
        
        if box_preds.dim() != 2 or box_preds.size(-1) != 4:
            print(f"Invalid box_preds shape: {box_preds.shape}")
            return False
        if target_boxes.dim() != 2 or target_boxes.size(-1) != 4:
            print(f"Invalid target_boxes shape: {target_boxes.shape}")
            return False
        return True

    def _print_debug(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def forward(self, images, targets=None):
        """Forward pass with proper error handling"""
        try:
            # Move inputs to correct device and ensure tensor types
            images = images.to(self.device, non_blocking=True)
            if targets:
                targets = [{
                    'boxes': t['boxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                    'labels': t['labels'].to(self.device, dtype=torch.long, non_blocking=True)
                } for t in targets]

            # Extract features
            x1 = self.layer1(images)  # [B, 128, H/4, W/4]
            x2 = self.layer2(x1)      # [B, 256, H/8, W/8]
            x3 = self.layer3(x2)      # [B, 512, H/16, W/16]

            # Simplified FPN pathway
            features = [x3, x2, x1]
            fpn_outs = []
            
            # Process each level
            for i, (feat, fpn_conv) in enumerate(zip(features, self.fpn_convs)):
                feat = fpn_conv(feat)  # Normalize channels to 256
                if i > 0:  # Upsample and add if not the first level
                    feat = feat + F.interpolate(fpn_outs[-1], size=feat.shape[-2:], mode='nearest')
                fpn_outs.append(feat)

            # Generate predictions
            box_preds_list = []
            cls_preds_list = []
            
            for feat, box_head, cls_head in zip(fpn_outs, self.box_head, self.cls_head):
                box_pred = box_head(feat)
                cls_pred = cls_head(feat)
                
                B, _, H, W = box_pred.shape
                box_preds_list.append(box_pred.permute(0, 2, 3, 1).reshape(B, -1, 4))
                cls_preds_list.append(cls_pred.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))

            # Concatenate predictions
            box_preds = torch.cat(box_preds_list, dim=1)
            cls_preds = torch.cat(cls_preds_list, dim=1)

            return [box_preds, cls_preds], {'boxes': box_preds, 'scores': cls_preds.sigmoid()}

        except Exception as e:
            print(f"Forward pass error: {str(e)}")
            print(traceback.format_exc())
            return None, None

    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Compute focal loss for classification"""
        # Convert inputs to probabilities
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.sum()

    def compute_loss(self, outputs, targets):
        """Compute loss with proper gradient handling"""
        try:
            box_preds, cls_preds = outputs
            batch_size = box_preds.size(0)
            device = box_preds.device
            batch_loss = torch.zeros(1, device=device, requires_grad=True)

            for idx in range(batch_size):
                # Ensure target tensors are properly formatted
                target_boxes = targets[idx]['boxes'].to(device, dtype=torch.float32)
                target_labels = targets[idx]['labels'].to(device, dtype=torch.long)
                
                if len(target_boxes) == 0:
                    continue

                # Calculate IoU and assign targets
                ious = self._box_iou(box_preds[idx], target_boxes)
                best_ious, matched_idx = ious.max(dim=1)
                pos_mask = best_ious > 0.5

                if pos_mask.any():
                    num_pos = pos_mask.sum().float().clamp(min=1)
                    
                    # Box regression loss
                    box_loss = F.smooth_l1_loss(
                        box_preds[idx][pos_mask],
                        target_boxes[matched_idx[pos_mask]],
                        reduction='sum'
                    ) / num_pos * self.box_weight
                    
                    # Classification loss
                    cls_target = torch.zeros_like(cls_preds[idx].squeeze(-1))
                    cls_target[pos_mask] = target_labels[matched_idx[pos_mask]].float()
                    cls_loss = F.binary_cross_entropy_with_logits(
                        cls_preds[idx].squeeze(-1),
                        cls_target,
                        reduction='sum'
                    ) / num_pos * self.cls_weight
                    
                    # Accumulate loss with gradient tracking
                    batch_loss = batch_loss + box_loss + cls_loss

            return batch_loss / batch_size

        except Exception as e:
            print(f"Loss computation error: {str(e)}")
            print(traceback.format_exc())
            return None

    def _box_iou(self, box1, box2):
        """Calculate IoU between boxes with gradient support"""
        # Convert to x1y1x2y2 format if needed
        if box1.size(-1) == 4:
            box1_x1y1 = box1[..., :2] - box1[..., 2:] / 2
            box1_x2y2 = box1[..., :2] + box1[..., 2:] / 2
            box1 = torch.cat([box1_x1y1, box1_x2y2], dim=-1)
        
        # Calculate intersection areas
        inter = (torch.min(box1[:, None, 2:], box2[None, :, 2:]) - 
                torch.max(box1[:, None, :2], box2[None, :, :2])).clamp(min=0)
        inter_area = inter.prod(dim=2)
        
        # Calculate union areas
        box1_area = ((box1[:, 2:] - box1[:, :2])).prod(dim=1)
        box2_area = ((box2[:, 2:] - box2[:, :2])).prod(dim=1)
        union = (box1_area[:, None] + box2_area[None, :] - inter_area)
        
        return inter_area / (union + 1e-6)

    def cuda(self, device=None):
        """Override cuda to handle memory efficiently"""
        device = device or self.device
        self.device = device
        super().cuda(device)
        # Pre-compute grids on GPU
        for size in self.grid_sizes:
            self._get_grid(*size, device)
        torch.cuda.empty_cache()
        return self
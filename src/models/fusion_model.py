
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN  # Ensure this matches your actual FasterRCNN import

class SafeFasterRCNN(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, images, targets=None):
        # Minimal safe target handling - focus on preserving valid detections
        if targets is None:
            targets = []

        # Ensure targets match image count
        targets = targets[:len(images)] + [{} for _ in range(max(0, len(images)-len(targets)))]

        # Basic target validation only
        safe_targets = []
        for img, t in zip(images, targets):
            device = img.device
            safe_t = {
                'boxes': torch.empty((0, 4), dtype=torch.float32, device=device),
                'labels': torch.empty(0, dtype=torch.int64, device=device)
            }

            if t and isinstance(t, dict) and 'boxes' in t and 'labels' in t:
                # Only basic checks - preserve most targets
                if isinstance(t['boxes'], torch.Tensor) and t['boxes'].shape[-1] == 4:
                    safe_t['boxes'] = t['boxes'].to(device).float()
                if isinstance(t['labels'], torch.Tensor) and len(t['labels']) == len(safe_t['boxes']):
                    safe_t['labels'] = t['labels'].to(device).long()

            safe_targets.append(safe_t)

        return self.model(images, safe_targets)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork
from collections import OrderedDict
import torchvision.models as models


# -------------------------- 1. Backbones with Enhanced Small Object Features --------------------------
class LightBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # Reduced stride for better small object features
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)  # [B, 64, H/4, W/4] (better resolution than before)


class ResNet18FPNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Use pretrained weights
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])

        # Include lower-level features (P2) for small objects
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],  # Added P2 (64ch) from layer1
            out_channels=256
        )

    def forward(self, x):
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # bn1
        x = self.backbone[2](x)  # relu
        x = self.backbone[3](x)  # maxpool
        f1 = self.backbone[4](x)  # Layer1 (64ch, stride=4) - NEW: P2 source
        f2 = self.backbone[5](f1)  # Layer2 (128ch, stride=8)
        f3 = self.backbone[6](f2)  # Layer3 (256ch, stride=16)
        f4 = self.backbone[7](f3)  # Layer4 (512ch, stride=32)

        # FPN with P2 (small objects) + P3 + P4 + P5
        fpn_feats = self.fpn(OrderedDict([('0', f1), ('1', f2), ('2', f3), ('3', f4)]))
        return list(fpn_feats.values())  # [P2, P3, P4, P5] (more scales)


# -------------------------- 2. Enhanced Adaptive Fusion Module --------------------------
class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, light_in_ch=64, fpn_in_ch=256, out_ch=256):
        super().__init__()
        self.light_conv = nn.Conv2d(light_in_ch, fpn_in_ch, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(fpn_in_ch)
        self.dropout = nn.Dropout(0.2)  # Reduced dropout for small objects

        # Add small object enhancement branch
        self.small_obj_branch = nn.Sequential(
            nn.Conv2d(fpn_in_ch, fpn_in_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_in_ch // 2, fpn_in_ch, kernel_size=3, padding=1)
        )

        # Stronger attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(fpn_in_ch * 2, fpn_in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_in_ch),  # Added batch norm for stability
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_in_ch, fpn_in_ch, kernel_size=3, padding=1),  # Deeper attention
            nn.BatchNorm2d(fpn_in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_in_ch, 2, kernel_size=1, padding=0),
            nn.Softmax(dim=1)
        )
        self.out_conv = nn.Conv2d(fpn_in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, light_feat, fpn_feat):
        light_feat_aligned = self.bn(self.light_conv(light_feat))

        # Enhance small object features
        small_obj_enhance = self.small_obj_branch(fpn_feat)
        fpn_feat = fpn_feat + 0.3 * small_obj_enhance  # Residual connection

        concat_feat = torch.cat([light_feat_aligned, fpn_feat], dim=1)
        weights = self.attention(concat_feat)
        fused_feat = weights[:, 0:1, :, :] * light_feat_aligned + weights[:, 1:2, :, :] * fpn_feat
        fused_feat = self.dropout(fused_feat)
        return self.out_conv(fused_feat)


# -------------------------- 3. Fused Backbone with More Scales --------------------------
class FusedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.light_backbone = LightBackbone()
        self.resnet_fpn = ResNet18FPNBackbone()
        # Add fusion for P2 (small objects)
        self.aff_p2 = AdaptiveFeatureFusion(light_in_ch=64, fpn_in_ch=256, out_ch=256)
        self.aff_p3 = AdaptiveFeatureFusion(light_in_ch=64, fpn_in_ch=256, out_ch=256)
        self.aff_p4 = AdaptiveFeatureFusion(light_in_ch=64, fpn_in_ch=256, out_ch=256)
        self.aff_p5 = AdaptiveFeatureFusion(light_in_ch=64, fpn_in_ch=256, out_ch=256)

        self.out_channels = 256

    def forward(self, x):
        light_feat = self.light_backbone(x)  # [B, 64, H/4, W/4] (higher resolution)
        p2, p3, p4, p5 = self.resnet_fpn(x)  # Include P2 for small objects

        # Upsample light features to match all FPN levels
        light_feat_p3 = F.interpolate(light_feat, size=p3.shape[2:], mode='bilinear', align_corners=True)
        light_feat_p4 = F.interpolate(light_feat, size=p4.shape[2:], mode='bilinear', align_corners=True)
        light_feat_p5 = F.interpolate(light_feat, size=p5.shape[2:], mode='bilinear', align_corners=True)

        # Fuse all levels (including P2 for small objects)
        p2_fused = self.aff_p2(light_feat, p2)  # P2: best for small cattle
        p3_fused = self.aff_p3(light_feat_p3, p3)
        p4_fused = self.aff_p4(light_feat_p4, p4)
        p5_fused = self.aff_p5(light_feat_p5, p5)

        return {'0': p2_fused, '1': p3_fused, '2': p4_fused, '3': p5_fused}


# -------------------------- 4. Detector with Recall-Focused Settings --------------------------
class AdaptiveFusionDetector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fused_backbone = FusedBackbone()

        # Anchor generator with sizes optimized for cattle
        self.anchor_generator = AnchorGenerator(
            sizes=((16, 32), (48, 64), (96, 128), (192, 256)),  # Smaller base sizes for distant cattle
            aspect_ratios=((0.5, 0.7, 1.0, 1.5, 2.0),) * 4  # More ratios to cover cattle shapes
        )

        self.roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # Include P2
            output_size=7,
            sampling_ratio=2
        )

        self.faster_rcnn = FasterRCNN(
            backbone=self.fused_backbone,
            num_classes=num_classes,
            rpn_anchor_generator=self.anchor_generator,
            box_roi_pool=self.roi_pooler,
            min_size=640,
            max_size=640,
            # RPN settings to capture more candidates
            rpn_pre_nms_top_n_train=3000,  # Down from 6000
            rpn_post_nms_top_n_train=2000, # Down from 4000
            rpn_pre_nms_top_n_test=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_fg_iou_thresh=0.3,
            rpn_bg_iou_thresh=0.05,
            rpn_batch_size_per_image=128,
            # Detection settings to reduce false negatives
            box_score_thresh=0.005,
            box_nms_thresh=0.4,
            box_detections_per_img=500,    # Down from 1000
            rpn_score_loss_fn=lambda obj_logits, obj_targets: F.binary_cross_entropy_with_logits(
                obj_logits, obj_targets, weight=torch.where(obj_targets==1, 5.0, 1.0)
            ),
        )

    def forward(self, x, targets=None):
        return self.faster_rcnn(x, targets)


def create_adaptive_fusion_model(num_classes: int) -> AdaptiveFusionDetector:
    # Return original FasterRCNN-based model (no wrapper)
    model = AdaptiveFusionDetector(num_classes=num_classes)
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model

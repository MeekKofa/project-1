
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork
from collections import OrderedDict
import torchvision.models as models

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
        # Enhanced multi-scale feature transformation
        self.light_conv = nn.Sequential(
            # Deep feature extraction for small objects
            nn.Conv2d(light_in_ch, light_in_ch * 2, 3, padding=1),
            nn.BatchNorm2d(light_in_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(light_in_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced FPN feature processing
        self.fpn_conv = nn.Sequential(
            nn.Conv2d(fpn_in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale feature aggregation
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch * 2, out_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch * 2, 1),
            nn.Sigmoid()
        )
        
        # Enhanced spatial attention for small objects
        self.spatial_att = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # More channels for better feature extraction
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 7, padding=3),  # Larger kernel for better context
            nn.Sigmoid()
        )
        
        # Scale-aware fusion
        self.scale_weights = nn.Parameter(torch.ones(2))  # Learnable scale weights
        self.smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # Final smoothing
        
    def forward(self, light_feat, fpn_feat):
        """Enhanced forward pass with multi-scale feature fusion."""
        # Transform features
        light_feat = self.light_conv(light_feat)
        fpn_feat = self.fpn_conv(fpn_feat)
        
        # Concatenate for attention
        cat_feat = torch.cat([light_feat, fpn_feat], dim=1)
        
        # Enhanced channel attention
        channel_weights = self.channel_att(cat_feat)
        light_channel = light_feat * channel_weights[:, :light_feat.size(1)]
        fpn_channel = fpn_feat * channel_weights[:, light_feat.size(1):]
        
        # Enhanced spatial attention with scale awareness
        spatial_feat = torch.cat([
            torch.mean(cat_feat, dim=1, keepdim=True),
            torch.max(cat_feat, dim=1, keepdim=True)[0],
            torch.std(cat_feat, dim=1, keepdim=True)  # Add std for better feature discrimination
        ], dim=1)
        spatial_weight = self.spatial_att(spatial_feat)
        
        # Scale-aware weighted fusion
        weights = F.softmax(self.scale_weights, dim=0)
        fused_feat = (
            weights[0] * light_channel * spatial_weight + 
            weights[1] * fpn_channel * spatial_weight
        )
        
        # Final feature refinement
        output = self.smooth(fused_feat)
        return output


class FeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),  # Group norm for better small batch training
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # 1x1 conv for channel adjustment
        )
        self.bn = nn.GroupNorm(8, out_channels)  # Group norm for better stability
        self.dropout = nn.Dropout(0.2)  # Reduced dropout for small objects

        # Small object enhancement branch
        self.small_obj_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        )

        # Multi-scale attention module with channel compression
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 2, kernel_size=1),
            nn.Sigmoid()
        )

        # Output convolution
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Main feature processing
        main_feat = self.conv(x)
        main_feat = self.bn(main_feat)
        main_feat = self.dropout(main_feat)
        
        # Small object enhancement
        small_feat = self.small_obj_branch(main_feat)
        
        # Concatenate features for attention
        combined = torch.cat([main_feat, small_feat], dim=1)
        attention_weights = self.attention(combined)
        
        # Apply attention weights
        enhanced_feat = (
            main_feat * attention_weights[:, 0:1] +
            small_feat * attention_weights[:, 1:2]
        )
        
        # Final output
        return self.out_conv(enhanced_feat)

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
        # Enhanced anchor generator with more size coverage
        self.anchor_generator = AnchorGenerator(
            sizes=((16, 24, 32), (48, 64, 96), (128, 192, 256), (320, 384, 512)),  # More granular sizes
            aspect_ratios=((0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0),) * 4  # More aspect ratios for better coverage
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
            # RPN settings optimized for recall
            rpn_pre_nms_top_n_train=6000,  # Increased to capture more potential objects
            rpn_post_nms_top_n_train=3000,  # Increased to keep more candidates
            rpn_pre_nms_top_n_test=3000,
            rpn_post_nms_top_n_test=1500,
            rpn_fg_iou_thresh=0.4,  # More lenient foreground threshold
            rpn_bg_iou_thresh=0.3,  # Higher background threshold
            rpn_batch_size_per_image=256,  # Larger batch size for better sampling
            rpn_positive_fraction=0.7,  # Higher fraction of positive samples
            # Detection settings optimized for recall
            box_score_thresh=0.05,  # Lower confidence threshold
            box_nms_thresh=0.45,  # Higher NMS threshold to keep more overlapping boxes
            box_detections_per_img=1000,  # Keep more detections
            box_fg_iou_thresh=0.4,  # More lenient foreground threshold for RoI
            box_bg_iou_thresh=0.3,  # Higher background threshold for RoI
            # Enhanced loss weighting for better object detection
            rpn_score_loss_fn=lambda obj_logits, obj_targets: F.binary_cross_entropy_with_logits(
                obj_logits, obj_targets, weight=torch.where(obj_targets==1, 8.0, 1.0)  # Higher weight on positive samples
            ),
            box_score_loss_fn=lambda cls_logits, cls_targets: F.cross_entropy(
                cls_logits, cls_targets, weight=torch.tensor([1.0, 2.0]).to(cls_logits.device)  # Class-balanced loss
            ),
        )

    def forward(self, x, targets=None):
        return self.faster_rcnn(x, targets)


def create_adaptive_fusion_model(num_classes: int) -> AdaptiveFusionDetector:
    model = AdaptiveFusionDetector(num_classes=num_classes)
    
    # Enhanced weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # MSRA initialization for better gradient flow
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Xavier initialization for fully connected layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Initialize RPN layers with bias towards foreground
    rpn = model.faster_rcnn.rpn.head
    for layer in rpn.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    # Initialize box predictor layers for better initial predictions
    box_predictor = model.faster_rcnn.roi_heads.box_predictor
    for name, param in box_predictor.named_parameters():
        if "weight" in name:
            nn.init.normal_(param, mean=0.0, std=0.01)
        elif "bias" in name:
            nn.init.zeros_(param)
            # Initialize classification bias to reduce initial false negatives
            if 'cls_score' in name:
                param.data.fill_(-2.0)
    
    return model

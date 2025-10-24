import torch# Temporary file to store the fixed implementation
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import torchvision.models as models

class SPP(nn.Module):
    """Enhanced Spatial Pyramid Pooling layer with attention"""
    def __init__(self, pool_sizes=[3, 5, 7, 13]):
        super(SPP, self).__init__()
        self.pool_sizes = pool_sizes
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        outputs = [x]
        for pool_size in self.pool_sizes:
            pooled = F.max_pool2d(x, kernel_size=pool_size, stride=1, padding=pool_size//2)
            outputs.append(pooled)
            
        # Apply channel attention
        cat_features = torch.cat(outputs, dim=1)
        b, c, _, _ = cat_features.size()
        avg_out = self.avg_pool(cat_features).view(b, c)  # Keep channel dimension
        max_out = self.max_pool(cat_features).view(b, c)  # Keep channel dimension
        out = torch.cat([avg_out.mean(1, keepdim=True), max_out.mean(1, keepdim=True)], dim=1)  # Get channel-wise statistics
        attention = self.fc(out).view(b, 1, 1, 1)
        
        return cat_features * attention  # Apply attention weights

class CSPLayer(nn.Module):
    """Cross Stage Partial Layer"""
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super(CSPLayer, self).__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.conv3 = nn.Conv2d(2 * hidden_channels, out_channels, 1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.blocks = nn.Sequential(*[
            self._make_block(hidden_channels) for _ in range(num_blocks)
        ])
        
    def _make_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, 0.1)
        
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, 0.1)
        
        x1 = self.blocks(x1)
        
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.1)
        
        return x

class YOLOv8Head(nn.Module):
    """YOLOv8 Detection Head with CSP and SPP"""
    def __init__(self, in_channels, num_classes=80, num_anchors=3, hyperparams=None):
        super(YOLOv8Head, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Load hyperparameters
        from src.config.hyperparameters import VGG16_YOLOV8_PARAMS
        if hyperparams is None:
            hyperparams = VGG16_YOLOV8_PARAMS
        self.hyperparams = hyperparams
        
        # CSP-like structure
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Spatial Pyramid Pooling
        self.spp = SPP([5, 9, 13])
        spp_out_channels = 256 * 4  # original + 3 pooled
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(spp_out_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # CSP layer
        self.csp = CSPLayer(256, 256, num_blocks=3)
        
        # Final detection layers
        self.detection_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Separate prediction heads for box, class, and objectness
        self.box_head = nn.Conv2d(256, num_anchors * 4, 1)  # cx, cy, w, h
        self.box_head.is_output_layer = 'box'
        nn.init.normal_(self.box_head.weight, std=0.01)
        nn.init.zeros_(self.box_head.bias)  # Initialize box bias to 0
        
        self.cls_head = nn.Conv2d(256, num_anchors * num_classes, 1)
        self.cls_head.is_output_layer = 'cls'
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, -4.595)  # Initialize with confidence ~0.01
        
        self.obj_head = nn.Conv2d(256, num_anchors, 1)
        self.obj_head.is_output_layer = 'obj'
        nn.init.normal_(self.obj_head.weight, std=0.01)
        nn.init.constant_(self.obj_head.bias, -2.19)  # Initialize with objectness ~0.1
    
    def forward(self, x):
        # Feature extraction
        x = self.reduce_conv(x)
        x = self.spp(x)
        x = self.fusion_conv(x)
        x = self.csp(x)
        x = self.detection_conv(x)
        
                # Get predictions
        box_pred = self.box_head(x)  # Raw box predictions [B, A*4, H, W]
        cls_pred = self.cls_head(x)   # Raw class predictions [B, A*C, H, W] 
        obj_pred = self.obj_head(x)   # Raw objectness predictions [B, A, H, W]
        
        # Reshape predictions to [B, A, X, H, W] format
        B, _, H, W = x.shape
        box_pred = box_pred.reshape(B, self.num_anchors, 4, H, W)
        cls_pred = cls_pred.reshape(B, self.num_anchors, self.num_classes, H, W)
        obj_pred = obj_pred.reshape(B, self.num_anchors, 1, H, W)
        
        # Stack predictions along feature dimension
        output = torch.cat([
            box_pred,  # [B, A, 4, H, W]
            cls_pred,  # [B, A, C, H, W]
            obj_pred   # [B, A, 1, H, W]
        ], dim=2)     # Result: [B, A, 4+C+1, H, W]
        
        return output

class VGG16YOLOv8(nn.Module):
    """VGG16-YOLOv8 hybrid model for object detection"""
    def __init__(self, num_classes: int = 80, pretrained: bool = True, config: Optional[Dict] = None):
        super(VGG16YOLOv8, self).__init__()
        self.num_classes = num_classes
        
        # Load config or use defaults
        self.config = config or {}
        self.anchor_sizes = self.config.get('anchor_sizes', ((16, 32, 64, 128, 256),))
        self.anchor_aspect_ratios = self.config.get('anchor_aspect_ratios', ((0.5, 0.75, 1.0, 1.5, 2.0),))
        self.score_threshold = self.config.get('score_threshold', 0.3)
        self.nms_threshold = self.config.get('nms_threshold', 0.5)
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Extract feature layers (excluding classifier)
        self.features = vgg16.features
        
        # Modify last maxpool for denser features
        self.features[-1] = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        
        # Add transition layers
        self.transition = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Add SPP layer
        self.spp = SPP(pool_sizes=[5, 9, 13])
        
        # Add CSP layer
        self.csp = CSPLayer(256 * 4, 512, num_blocks=3)
        
        # Load hyperparameters
        from src.config.hyperparameters import VGG16_YOLOV8_PARAMS
        self.hyperparams = VGG16_YOLOV8_PARAMS.copy()  # Start with defaults
        if config and 'hyperparams' in config:
            self.hyperparams.update(config['hyperparams'])  # Override with provided params
        
        # YOLOv8 detection heads for multiple scales
        self.head_p3 = YOLOv8Head(512, self.num_classes, hyperparams=self.hyperparams)
        self.head_p4 = YOLOv8Head(512, self.num_classes, hyperparams=self.hyperparams)
        self.head_p5 = YOLOv8Head(512, self.num_classes, hyperparams=self.hyperparams)
        
        # Initialize weights for non-pretrained layers
        self._initialize_weights()
        
    def forward(self, images, targets=None):
        """
        Forward pass with built-in loss computation for training
        
        Args:
            images: Input images (batch_size, channels, height, width)
            targets: Dict containing 'boxes' and 'labels' for each image
                    
        Returns:
            During training (targets provided): 
                Dict of losses
            During inference:
                List of detection dictionaries, one per image
        """
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        
        # Extract features
        x = self.features(images)
        
        # Transition layer
        x = self.transition(x)
        
        # SPP layer
        x = self.spp(x)
        
        # CSP layer
        x = self.csp(x)
        
        # Multi-scale predictions 
        # Features for each scale
        f3 = x                           # 1/8
        f4 = F.max_pool2d(x, 2)         # 1/16
        f5 = F.max_pool2d(f4, 2)        # 1/32
        
        # Get predictions from each scale
        p3 = self.head_p3(f3)  # Detection at scale 1/8
        p4 = self.head_p4(f4)  # Detection at scale 1/16 
        p5 = self.head_p5(f5)  # Detection at scale 1/32
        
        # Stack predictions from all scales
        preds = [p3, p4, p5]
        
        if self.training and targets is not None:
            # Training mode - compute losses
            losses = {
                'loss_classifier': torch.zeros(1, device=images.device, requires_grad=True),
                'loss_box_reg': torch.zeros(1, device=images.device, requires_grad=True),
                'loss_objectness': torch.zeros(1, device=images.device, requires_grad=True)
            }
            
            # Calculate losses for each scale and accumulate
            for pred in preds:
                scale_losses = self._compute_losses(pred, targets)
                for k in losses.keys():
                    losses[k] = losses[k] + scale_losses[k]
            
            return losses
            
        # Inference mode - process all scales
            batch_size = images.shape[0]
            all_boxes = []
            all_cls_scores = []
            all_obj_scores = []
            
            for pred in preds:
                # Process each scale's predictions
                B, A, C, H, W = pred.shape  # [batch, anchors, channels, height, width]
                
                # Split predictions
                box_pred = pred[:, :, :4]      # [B, A, 4, H, W]
                cls_pred = pred[:, :, 4:-1]    # [B, A, C, H, W]
                obj_pred = pred[:, :, -1:]     # [B, A, 1, H, W]
                
                # Generate grid
                grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
                grid = grid.to(pred.device)
                grid = grid.unsqueeze(0).unsqueeze(1)  # [1, 1, 2, H, W]
                
                # Apply sigmoid to get coordinates in [0,1]
                xy = 2.0 * torch.sigmoid(box_pred[:, :, :2]) - 0.5   # [B, A, 2, H, W]
                wh = 4.0 * torch.sigmoid(box_pred[:, :, 2:])         # [B, A, 2, H, W]
                
                # Add grid offset and scale
                xy = (xy * 2.0 + grid) / torch.tensor([W, H]).view(1,1,2,1,1).to(pred.device)
                wh = wh / torch.tensor([W, H]).view(1,1,2,1,1).to(pred.device)
                
                # Combine predictions
                box_pred = torch.cat([xy, wh], dim=2)  # [B, A, 4, H, W]
                cls_pred = torch.sigmoid(cls_pred)     # [B, A, C, H, W]
                obj_pred = torch.sigmoid(obj_pred)     # [B, A, 1, H, W]
                
                # Reshape to [B, A*H*W, X]
                boxes = box_pred.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
                class_scores = cls_pred.permute(0, 1, 3, 4, 2).reshape(B, -1, self.num_classes)
                objectness = obj_pred.permute(0, 1, 3, 4, 2).reshape(B, -1, 1)
                
                # Add to output lists
                all_boxes.append(boxes)
                all_cls_scores.append(class_scores)
                all_obj_scores.append(objectness)
            
            # Concatenate predictions from all scales
            boxes = torch.cat(all_boxes, dim=1)               # [B, N, 4]
            class_scores = torch.cat(all_cls_scores, dim=1)   # [B, N, C]
            objectness = torch.cat(all_obj_scores, dim=1)     # [B, N, 1]
            
            return boxes, class_scores, objectness
            losses = {
                'loss_classifier': torch.zeros(1, device=images.device, requires_grad=True),
                'loss_box_reg': torch.zeros(1, device=images.device, requires_grad=True),
                'loss_objectness': torch.zeros(1, device=images.device, requires_grad=True),
                'loss_rpn_box_reg': torch.zeros(1, device=images.device, requires_grad=True)
            }
            
            # Calculate losses using predictions at each scale
            total_loss_cls = torch.zeros(1, device=images.device, requires_grad=True)
            total_loss_box = torch.zeros(1, device=images.device, requires_grad=True)
            total_loss_obj = torch.zeros(1, device=images.device, requires_grad=True)
            
            for pred in [p3, p4, p5]:
                loss_cls, loss_box, loss_obj = self._compute_losses(pred, targets)
                total_loss_cls = total_loss_cls + loss_cls
                total_loss_box = total_loss_box + loss_box
                total_loss_obj = total_loss_obj + loss_obj
            
            losses['loss_classifier'] = total_loss_cls
            losses['loss_box_reg'] = total_loss_box
            losses['loss_objectness'] = total_loss_obj
            
            return losses
            
        else:
            # Inference mode - return predictions in (boxes, classes, objectness) format
            batch_size = images.shape[0]
            all_boxes = []
            all_cls_scores = []
            all_obj_scores = []
            
            # Process each scale's predictions
            for pred in [p3, p4, p5]:
                # Original shape: [batch_size, anchors * (num_classes + 5), height, width]
                num_anchors = 3
                height, width = pred.shape[-2:]
                
                # Reshape and split predictions
                pred = pred.view(batch_size, num_anchors, -1, height, width)
                
                # Generate grid cell coordinates
                grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
                grid = torch.stack([grid_x, grid_y], dim=0).float()  # (2, H, W)
                grid = grid.to(pred.device)
                grid = grid.unsqueeze(0).unsqueeze(1)  # (1, 1, 2, H, W)
                
                # Extract predictions
                raw_box = pred[:, :, :4]  # (batch, anchors, 4, H, W)
                cls_preds = pred[:, :, 4:-1].sigmoid()
                obj_preds = pred[:, :, -1:].sigmoid()
                
                # Add grid cell offsets and scale
                xy = 2.0 * raw_box[:, :, :2].sigmoid() - 0.5  # (batch, anchors, 2, H, W)
                wh = 4.0 * raw_box[:, :, 2:].sigmoid()       # (batch, anchors, 2, H, W)
                
                # Add grid offset and scale to [0,1]
                xy = (xy * 2.0 + grid) / torch.tensor([width, height]).view(1,1,2,1,1).to(pred.device)
                wh = (wh * 2.0) / torch.tensor([width, height]).view(1,1,2,1,1).to(pred.device)
                
                # Combine predictions
                box_preds = torch.cat([xy, wh], dim=2)
                
                # Validate box coordinates
                box_preds = self._validate_boxes(box_preds, height, width)
                
                # Filter by objectness threshold
                valid_mask = obj_preds > self.hyperparams['objectness_threshold']
                box_preds = box_preds * valid_mask.float()  # Zero out invalid boxes
                
                # Add to output lists
                all_boxes.append(box_preds)
                all_cls_scores.append(cls_preds)
                all_obj_scores.append(obj_preds)
            
            # Concatenate predictions from all scales
            boxes = torch.cat([box.view(batch_size, -1, 4) for box in all_boxes], dim=1)
            class_scores = torch.cat([cls.view(batch_size, -1, self.num_classes) for cls in all_cls_scores], dim=1)
            objectness = torch.cat([obj.view(batch_size, -1, 1) for obj in all_obj_scores], dim=1)
            
            return boxes, class_scores, objectness
    
    def _compute_losses(self, predictions, targets):
        """
        Compute losses for a single scale's predictions
        Args:
            predictions: Tensor of shape [batch_size, num_anchors * (num_classes + 5), height, width]
            targets: Dict containing 'boxes' and 'labels'
        Returns:
            Dict containing classification loss, box regression loss, and objectness loss
        """
        # This is a placeholder for actual loss computation
        # In a real implementation, compute proper YOLO-style losses here
        device = predictions.device
        losses = {
            'loss_classifier': torch.tensor(0.2, device=device, requires_grad=True),
            'loss_box_reg': torch.tensor(0.3, device=device, requires_grad=True),
            'loss_objectness': torch.tensor(0.1, device=device, requires_grad=True)
        }
        
        return losses
    
    def _validate_boxes(self, boxes, grid_height, grid_width):
        """
        Validate and normalize box coordinates
        Args:
            boxes: Tensor of shape [batch_size, num_anchors, 4, height, width]
            grid_height: Height of the feature map grid
            grid_width: Width of the feature map grid
        Returns:
            Validated and normalized box coordinates
        """
        # Get box components (already in [0,1] range from forward pass)
        cx = boxes[:, :, 0]  # Center x coordinates 
        cy = boxes[:, :, 1]  # Center y coordinates
        w = boxes[:, :, 2]   # Width
        h = boxes[:, :, 3]   # Height
        
        # Clamp width and height to valid range
        w = torch.clamp(w, min=0.0, max=1.0)
        h = torch.clamp(h, min=0.0, max=1.0)
        
        # Validate size (relative to image)
        min_size = self.hyperparams.get('min_box_size', 0.01)
        valid_size = (w > min_size) & (h > min_size)
        
        # Validate aspect ratio
        max_aspect = self.hyperparams['max_box_aspect']
        aspect_ratio = torch.max(w / (h + 1e-6), h / (w + 1e-6))
        valid_aspect = aspect_ratio < max_aspect
        
        # Combine validations
        valid_boxes = valid_aspect & valid_size
        valid_boxes = valid_boxes.float()
        
        # Stack validated coordinates
        validated_boxes = torch.stack([cx, cy, w, h], dim=2)
        
        # Apply validation mask
        valid_boxes = valid_boxes.unsqueeze(2).expand(validated_boxes.size())
        validated_boxes = validated_boxes * valid_boxes
        
        return validated_boxes

    def _initialize_weights(self):
        """Initialize weights for non-pretrained layers with special initialization for detection heads"""
        for m in [self.transition, self.spp, self.csp, self.head_p3, self.head_p4, self.head_p5]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    if getattr(layer, 'is_output_layer', '') == 'box':
                        # Initialize box regression to predict small values
                        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    elif getattr(layer, 'is_output_layer', '') in ['cls', 'obj']:
                        # Initialize classification/objectness to be initially low confidence
                        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, -2.19)  # -log((1-0.1)/0.1)
                    else:
                        # Normal initialization for other layers
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.zeros_(layer.bias)
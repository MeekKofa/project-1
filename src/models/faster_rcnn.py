# models/faster_rcnn.py
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class LightBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)  # single feature map (Tensor)


def create_cattle_detection_model(num_classes: int):
    backbone = LightBackbone()
    backbone.out_channels = 64  # REQUIRED by FasterRCNN

    # ONE feature map → ONE tuple of sizes / aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # For a single-tensor backbone, torchvision wraps it as {'0': tensor}
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # Let the model handle resizing/normalization; fixed size is fine
        min_size=384,
        max_size=384,
        rpn_pre_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_train=500,
        rpn_post_nms_top_n_test=250,
        rpn_batch_size_per_image=32,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        box_detections_per_img=100,
        box_score_thresh=0.01,
        box_nms_thresh=0.3,
    )
    return model

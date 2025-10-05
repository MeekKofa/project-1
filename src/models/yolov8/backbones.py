"""
YOLOv8 Backbone Options.

Provides multiple backbone choices:
1. Custom CSP-based (original YOLOv8)
2. ResNet-based (hybrid ResNet + YOLOv8)
3. ResNeXt-based (for better performance)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = 'silu'
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross Stage Partial Block for efficient feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3):
        super().__init__()
        hidden_channels = out_channels // 2

        # Split convolution
        self.conv1 = ConvBlock(
            in_channels, hidden_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(
            in_channels, hidden_channels, kernel_size=1, padding=0)

        # Bottleneck blocks
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvBlock(hidden_channels, hidden_channels,
                          kernel_size=1, padding=0),
                ConvBlock(hidden_channels, hidden_channels,
                          kernel_size=3, padding=1)
            )
            for _ in range(num_blocks)
        ])

        # Fusion convolution
        self.conv3 = ConvBlock(hidden_channels * 2,
                               out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split path
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        # Process through blocks
        x2 = self.blocks(x2)

        # Concatenate and fuse
        out = torch.cat([x1, x2], dim=1)
        out = self.conv3(out)

        return out


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast version."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2

        self.conv1 = ConvBlock(
            in_channels, hidden_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(hidden_channels * 4,
                               out_channels, kernel_size=1, padding=0)
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        # Multi-scale pooling
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)

        # Concatenate all scales
        out = torch.cat([x, y1, y2, y3], dim=1)
        out = self.conv2(out)

        return out


class CSPBackbone(nn.Module):
    """Original YOLOv8 CSP-based Backbone."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        self.name = "CSP"

        # Stem
        self.stem = ConvBlock(in_channels, base_channels,
                              kernel_size=3, stride=2, padding=1)

        # Stage 1: 64 -> 128
        self.stage1 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            CSPBlock(base_channels * 2, base_channels * 2, num_blocks=3)
        )

        # Stage 2: 128 -> 256
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4,
                      kernel_size=3, stride=2, padding=1),
            CSPBlock(base_channels * 4, base_channels * 4, num_blocks=6)
        )

        # Stage 3: 256 -> 512
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8,
                      kernel_size=3, stride=2, padding=1),
            CSPBlock(base_channels * 8, base_channels * 8, num_blocks=6)
        )

        # Stage 4: 512 -> 512 with SPPF
        self.stage4 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 8,
                      kernel_size=3, stride=2, padding=1),
            CSPBlock(base_channels * 8, base_channels * 8, num_blocks=3),
            SPPF(base_channels * 8, base_channels * 8)
        )

        self.out_channels = [base_channels * 4,
                             base_channels * 8, base_channels * 8]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through backbone.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            List of feature maps at different scales [c2, c3, c4]
        """
        x = self.stem(x)

        c1 = self.stage1(x)   # /4
        c2 = self.stage2(c1)  # /8  - 256 channels
        c3 = self.stage3(c2)  # /16 - 512 channels
        c4 = self.stage4(c3)  # /32 - 512 channels

        return [c2, c3, c4]  # Return multi-scale features


class ResNetBackbone(nn.Module):
    """
    ResNet-based Backbone for YOLOv8.

    Combines pretrained ResNet with YOLOv8 detection heads.
    Options: ResNet18, ResNet34, ResNet50, ResNet101
    """

    def __init__(
        self,
        resnet_type: str = 'resnet50',
        pretrained: bool = True,
        in_channels: int = 3
    ):
        super().__init__()

        self.name = f"ResNet-{resnet_type}"

        # Load ResNet backbone
        if resnet_type == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            feature_channels = [128, 256, 512]
        elif resnet_type == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            feature_channels = [128, 256, 512]
        elif resnet_type == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            feature_channels = [512, 1024, 2048]
        elif resnet_type == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            feature_channels = [512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Extract layers (conv1 -> layer4)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # /4
        self.layer2 = backbone.layer2  # /8
        self.layer3 = backbone.layer3  # /16
        self.layer4 = backbone.layer4  # /32

        # Add SPPF after layer4 for multi-scale features
        self.sppf = SPPF(feature_channels[2], feature_channels[2])

        self.out_channels = feature_channels

        logger.info(
            f"✅ Loaded {resnet_type} backbone (pretrained={pretrained})")
        logger.info(f"   Output channels: {feature_channels}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through ResNet backbone.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            List of feature maps at different scales [c2, c3, c4]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        c1 = self.layer1(x)   # /4
        c2 = self.layer2(c1)  # /8  - For small objects
        c3 = self.layer3(c2)  # /16 - For medium objects
        c4 = self.layer4(c3)  # /32 - For large objects

        # Apply SPPF to final layer
        c4 = self.sppf(c4)

        return [c2, c3, c4]  # Return multi-scale features


class ResNeXtBackbone(nn.Module):
    """
    ResNeXt-based Backbone for YOLOv8.

    Better than ResNet for complex scenes.
    Options: ResNeXt50, ResNeXt101
    """

    def __init__(
        self,
        resnext_type: str = 'resnext50_32x4d',
        pretrained: bool = True,
        in_channels: int = 3
    ):
        super().__init__()

        self.name = f"ResNeXt-{resnext_type}"

        # Load ResNeXt backbone
        if resnext_type == 'resnext50_32x4d':
            backbone = models.resnext50_32x4d(pretrained=pretrained)
            feature_channels = [512, 1024, 2048]
        elif resnext_type == 'resnext101_32x8d':
            backbone = models.resnext101_32x8d(pretrained=pretrained)
            feature_channels = [512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNeXt type: {resnext_type}")

        # Extract layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # /4
        self.layer2 = backbone.layer2  # /8
        self.layer3 = backbone.layer3  # /16
        self.layer4 = backbone.layer4  # /32

        # Add SPPF
        self.sppf = SPPF(feature_channels[2], feature_channels[2])

        self.out_channels = feature_channels

        logger.info(
            f"✅ Loaded {resnext_type} backbone (pretrained={pretrained})")
        logger.info(f"   Output channels: {feature_channels}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through ResNeXt backbone."""
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNeXt stages
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Apply SPPF
        c4 = self.sppf(c4)

        return [c2, c3, c4]


def build_backbone(
    backbone_type: str = 'csp',
    pretrained: bool = True,
    in_channels: int = 3,
    base_channels: int = 64
) -> nn.Module:
    """
    Factory function to build backbone.

    Args:
        backbone_type: Type of backbone
            - 'csp': Original YOLOv8 CSP backbone
            - 'resnet18', 'resnet34', 'resnet50', 'resnet101': ResNet backbones
            - 'resnext50_32x4d', 'resnext101_32x8d': ResNeXt backbones
        pretrained: Whether to use pretrained weights (for ResNet/ResNeXt)
        in_channels: Number of input channels
        base_channels: Base number of channels (for CSP backbone)

    Returns:
        Backbone module
    """
    backbone_type = backbone_type.lower()

    if backbone_type == 'csp':
        return CSPBackbone(in_channels=in_channels, base_channels=base_channels)

    elif backbone_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        return ResNetBackbone(resnet_type=backbone_type, pretrained=pretrained, in_channels=in_channels)

    elif backbone_type in ['resnext50_32x4d', 'resnext101_32x8d']:
        return ResNeXtBackbone(resnext_type=backbone_type, pretrained=pretrained, in_channels=in_channels)

    else:
        raise ValueError(
            f"Unsupported backbone type: {backbone_type}. "
            f"Choose from: csp, resnet18, resnet34, resnet50, resnet101, "
            f"resnext50_32x4d, resnext101_32x8d"
        )


__all__ = [
    'CSPBackbone',
    'ResNetBackbone',
    'ResNeXtBackbone',
    'build_backbone',
    'ConvBlock',
    'CSPBlock',
    'SPPF'
]

"""
编码器模块 - 超声分支和白光分支

超声分支输入: 3通道 (灰度 + 边缘/梯度 + 深度/ROI mask)
白光分支输入: 3通道 (RGB)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights, ResNet18_Weights,
    ConvNeXt_Tiny_Weights,
    ViT_B_16_Weights
)
from typing import Optional


class BaseEncoder(nn.Module):
    """
    基础编码器类

    支持的backbone:
    - resnet50, resnet18
    - convnext_tiny
    - vit_b_16
    """

    BACKBONE_OUTPUT_DIMS = {
        'resnet50': 2048,
        'resnet18': 512,
        'convnext_tiny': 768,
        'vit_b_16': 768,
    }

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 512,
        input_channels: int = 3,
        freeze_early_layers: bool = True,
        freeze_until_layer: int = 2
    ):
        super().__init__()

        self.backbone_name = backbone
        self.feature_dim = feature_dim

        # 创建backbone
        self.backbone, backbone_dim = self._create_backbone(
            backbone, pretrained, input_channels
        )

        # 冻结早期层
        if freeze_early_layers:
            self._freeze_layers(freeze_until_layer)

        # 投影层: 将backbone输出映射到统一维度
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def _create_backbone(
        self,
        backbone: str,
        pretrained: bool,
        input_channels: int
    ):
        """创建backbone网络"""

        if backbone == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            # 修改第一层以适应不同输入通道
            if input_channels != 3:
                model.conv1 = self._adapt_first_conv(
                    model.conv1, input_channels
                )
            # 移除分类层
            model.fc = nn.Identity()
            backbone_dim = 2048

        elif backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            if input_channels != 3:
                model.conv1 = self._adapt_first_conv(
                    model.conv1, input_channels
                )
            model.fc = nn.Identity()
            backbone_dim = 512

        elif backbone == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            if input_channels != 3:
                # ConvNeXt第一层
                old_conv = model.features[0][0]
                model.features[0][0] = self._adapt_first_conv(
                    old_conv, input_channels
                )
            model.classifier = nn.Identity()
            backbone_dim = 768

        elif backbone == 'vit_b_16':
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
            if input_channels != 3:
                # ViT patch embedding
                old_conv = model.conv_proj
                model.conv_proj = nn.Conv2d(
                    input_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding
                )
                if pretrained:
                    # 复制权重并调整
                    with torch.no_grad():
                        if input_channels < 3:
                            model.conv_proj.weight.copy_(
                                old_conv.weight[:, :input_channels]
                            )
                        else:
                            model.conv_proj.weight[:, :3].copy_(old_conv.weight)
                            # 额外通道用平均初始化
                            for i in range(3, input_channels):
                                model.conv_proj.weight[:, i].copy_(
                                    old_conv.weight.mean(dim=1)
                                )
            model.heads = nn.Identity()
            backbone_dim = 768

        else:
            raise ValueError(f"不支持的backbone: {backbone}")

        return model, backbone_dim

    def _adapt_first_conv(
        self,
        old_conv: nn.Conv2d,
        new_channels: int
    ) -> nn.Conv2d:
        """调整第一个卷积层以适应不同输入通道"""
        new_conv = nn.Conv2d(
            new_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # 复制预训练权重
        with torch.no_grad():
            if new_channels <= 3:
                new_conv.weight.copy_(old_conv.weight[:, :new_channels])
            else:
                # 前3个通道使用原始权重
                new_conv.weight[:, :3].copy_(old_conv.weight)
                # 额外通道使用平均值初始化
                for i in range(3, new_channels):
                    new_conv.weight[:, i].copy_(old_conv.weight.mean(dim=1))

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        return new_conv

    def _freeze_layers(self, freeze_until: int):
        """冻结早期层"""
        if 'resnet' in self.backbone_name:
            # ResNet: conv1, bn1, layer1, layer2, layer3, layer4
            layers_to_freeze = []
            if freeze_until >= 0:
                layers_to_freeze.extend([
                    self.backbone.conv1, self.backbone.bn1
                ])
            if freeze_until >= 1:
                layers_to_freeze.append(self.backbone.layer1)
            if freeze_until >= 2:
                layers_to_freeze.append(self.backbone.layer2)
            if freeze_until >= 3:
                layers_to_freeze.append(self.backbone.layer3)

            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        elif 'convnext' in self.backbone_name:
            # ConvNeXt: features[0-7]
            for i in range(min(freeze_until * 2, 6)):
                for param in self.backbone.features[i].parameters():
                    param.requires_grad = False

        elif 'vit' in self.backbone_name:
            # ViT: 冻结patch embedding和前几个transformer块
            for param in self.backbone.conv_proj.parameters():
                param.requires_grad = False
            for i in range(min(freeze_until * 4, 8)):
                for param in self.backbone.encoder.layers[i].parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            特征向量 [B, feature_dim]
        """
        # 提取backbone特征
        features = self.backbone(x)

        # 投影到统一维度
        features = self.projection(features)

        return features


class UltrasoundEncoder(BaseEncoder):
    """
    超声编码器

    输入: N通道超声图像 (由数据预处理决定通道数)
    输出: 帧级特征向量
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 512,
        input_channels: int = 3,
        freeze_early_layers: bool = True,
        freeze_until_layer: int = 2
    ):
        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            input_channels=input_channels,
            freeze_early_layers=freeze_early_layers,
            freeze_until_layer=freeze_until_layer
        )


class WhiteLightEncoder(BaseEncoder):
    """
    白光编码器

    输入: RGB白光内镜图像
    输出: 帧级特征向量
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 512,
        freeze_early_layers: bool = True,
        freeze_until_layer: int = 2
    ):
        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            input_channels=3,  # RGB
            freeze_early_layers=freeze_early_layers,
            freeze_until_layer=freeze_until_layer
        )


def create_encoder(
    modality: str,
    backbone: str = 'resnet50',
    pretrained: bool = True,
    feature_dim: int = 512,
    freeze_early_layers: bool = True,
    freeze_until_layer: int = 2
) -> BaseEncoder:
    """
    创建编码器的工厂函数

    Args:
        modality: 'ultrasound' 或 'white_light'
        backbone: backbone类型
        pretrained: 是否使用预训练权重
        feature_dim: 输出特征维度
        freeze_early_layers: 是否冻结早期层
        freeze_until_layer: 冻结到哪一层

    Returns:
        编码器实例
    """
    if modality == 'ultrasound':
        return UltrasoundEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            freeze_early_layers=freeze_early_layers,
            freeze_until_layer=freeze_until_layer
        )
    elif modality == 'white_light':
        return WhiteLightEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            freeze_early_layers=freeze_early_layers,
            freeze_until_layer=freeze_until_layer
        )
    else:
        raise ValueError(f"未知的modality: {modality}")

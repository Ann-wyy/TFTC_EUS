"""
EUS (Endoscopic Ultrasound) Encoder Module.

This module implements the EUS image encoder using CNN-based models
(ConvNeXt or ResNet) for extracting features from endoscopic ultrasound images.
"""

import torch
import torch.nn as nn
from typing import Optional
import torchvision.models as models
from torchvision.models import (
    ConvNeXt_Base_Weights, ConvNeXt_Small_Weights,
    ResNet50_Weights, ResNet101_Weights
)


class EUSEncoder(nn.Module):
    """
    EUS Image Encoder using CNN-based models.

    Supports:
    - ConvNeXt (Base/Small)
    - ResNet (50/101)

    CNN architectures are well-suited for EUS images as they can effectively
    capture:
    - Ultrasound echo textures and speckle patterns
    - Layered structure information
    - Boundary and edge features

    Args:
        model_name: Name of the backbone model. Options:
            - 'convnext_base': ConvNeXt Base
            - 'convnext_small': ConvNeXt Small
            - 'resnet50': ResNet-50
            - 'resnet101': ResNet-101
        pretrained: Whether to use pretrained weights from ImageNet.
        output_dim: Desired output feature dimension.
        freeze_layers: Number of layers/stages to freeze for transfer learning.
    """

    SUPPORTED_MODELS = {
        'convnext_base': (models.convnext_base, ConvNeXt_Base_Weights.DEFAULT, 1024),
        'convnext_small': (models.convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
        'resnet50': (models.resnet50, ResNet50_Weights.DEFAULT, 2048),
        'resnet101': (models.resnet101, ResNet101_Weights.DEFAULT, 2048),
    }

    def __init__(
        self,
        model_name: str = 'convnext_base',
        pretrained: bool = True,
        output_dim: int = 1024,
        freeze_layers: int = 0
    ):
        super().__init__()

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        model_fn, weights, backbone_dim = self.SUPPORTED_MODELS[model_name]
        self.model_name = model_name
        self.backbone_dim = backbone_dim
        self.output_dim = output_dim

        # Load the backbone model
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)

        # Remove the classification head
        self._remove_classification_head()

        # Freeze layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Global average pooling (for ResNet, ConvNeXt has its own)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection layer to desired output dimension
        if backbone_dim != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(backbone_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        else:
            self.projection = nn.Identity()

    def _remove_classification_head(self):
        """Remove the classification head from the backbone."""
        if 'convnext' in self.model_name:
            # ConvNeXt: remove classifier
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in self.model_name:
            # ResNet: remove fc layer
            self.backbone.fc = nn.Identity()

    def _freeze_layers(self, num_layers: int):
        """Freeze the first num_layers stages of the backbone."""
        if 'convnext' in self.model_name:
            # ConvNeXt: freeze first N stages
            # features: [0]=stem, [1,3,5,7]=stages, [2,4,6]=downsampling
            stages_to_freeze = min(num_layers, 4)
            for i in range(stages_to_freeze * 2):
                for param in self.backbone.features[i].parameters():
                    param.requires_grad = False
        elif 'resnet' in self.model_name:
            # ResNet: freeze conv1, bn1, and first N layers
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False

            layers = [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4
            ]
            for i in range(min(num_layers, 4)):
                for param in layers[i].parameters():
                    param.requires_grad = False

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone."""
        if 'convnext' in self.model_name:
            # ConvNeXt forward through features
            features = self.backbone.features(x)
            features = self.backbone.avgpool(features)
            features = features.flatten(1)
        elif 'resnet' in self.model_name:
            # ResNet forward
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            features = self.global_pool(x)
            features = features.flatten(1)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the EUS encoder.

        Args:
            x: Input EUS image tensor of shape (B, C, H, W).
               Expected: C=1 (grayscale) or C=3 (RGB), H=W=224.
               Note: If grayscale, will be converted to 3 channels.

        Returns:
            Feature tensor of shape (B, output_dim).
        """
        # Handle grayscale images by repeating channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Extract features
        features = self._forward_features(x)

        # Project to output dimension
        features = self.projection(features)

        return features

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.output_dim

    @property
    def num_parameters(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EUSEncoderWithSPP(EUSEncoder):
    """
    EUS Encoder with Spatial Pyramid Pooling for multi-scale feature extraction.

    This variant adds spatial pyramid pooling to capture ultrasound features
    at multiple scales, which is particularly useful for EUS images where
    tumor boundaries and textures may appear at different scales.
    """

    def __init__(
        self,
        model_name: str = 'convnext_base',
        pretrained: bool = True,
        output_dim: int = 1024,
        freeze_layers: int = 0,
        pool_sizes: tuple = (1, 2, 4)
    ):
        super().__init__(model_name, pretrained, output_dim, freeze_layers)

        self.pool_sizes = pool_sizes

        # Calculate SPP output dimension
        spp_dim = self.backbone_dim * sum(s * s for s in pool_sizes)

        # SPP pooling layers
        self.spp_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((s, s)) for s in pool_sizes
        ])

        # Replace projection with one that handles SPP output
        self.projection = nn.Sequential(
            nn.Linear(spp_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with spatial pyramid pooling."""
        if 'convnext' in self.model_name:
            features = self.backbone.features(x)
        elif 'resnet' in self.model_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            features = self.backbone.layer4(x)

        # Apply SPP
        batch_size = features.shape[0]
        spp_features = []
        for pool in self.spp_pools:
            pooled = pool(features)
            spp_features.append(pooled.flatten(1))

        # Concatenate all scales
        return torch.cat(spp_features, dim=1)


class EUSEncoderWithSEBlock(EUSEncoder):
    """
    EUS Encoder with Squeeze-and-Excitation attention for enhanced
    channel-wise feature recalibration.

    SE blocks help the model focus on the most informative channels
    for EUS image analysis.
    """

    def __init__(
        self,
        model_name: str = 'convnext_base',
        pretrained: bool = True,
        output_dim: int = 1024,
        freeze_layers: int = 0,
        se_reduction: int = 16
    ):
        super().__init__(model_name, pretrained, output_dim, freeze_layers)

        # SE Block
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone_dim, self.backbone_dim // se_reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone_dim // se_reduction, self.backbone_dim),
            nn.Sigmoid()
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with SE attention."""
        if 'convnext' in self.model_name:
            features = self.backbone.features(x)
        elif 'resnet' in self.model_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            features = self.backbone.layer4(x)

        # Apply SE attention
        se_weights = self.se_block(features)
        se_weights = se_weights.unsqueeze(-1).unsqueeze(-1)
        features = features * se_weights

        # Global pooling
        features = self.global_pool(features).flatten(1)

        return features


def create_eus_encoder(
    model_name: str = 'convnext_base',
    pretrained: bool = True,
    output_dim: int = 1024,
    freeze_layers: int = 0,
    variant: str = 'default'
) -> EUSEncoder:
    """
    Factory function to create EUS encoder.

    Args:
        model_name: Backbone model name.
        pretrained: Use pretrained weights.
        output_dim: Output feature dimension.
        freeze_layers: Number of layers to freeze.
        variant: Model variant. Options:
            - 'default': Standard EUS encoder
            - 'spp': With spatial pyramid pooling
            - 'se': With squeeze-and-excitation attention

    Returns:
        EUSEncoder instance.
    """
    if variant == 'spp':
        return EUSEncoderWithSPP(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            freeze_layers=freeze_layers
        )
    elif variant == 'se':
        return EUSEncoderWithSEBlock(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            freeze_layers=freeze_layers
        )
    else:
        return EUSEncoder(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            freeze_layers=freeze_layers
        )

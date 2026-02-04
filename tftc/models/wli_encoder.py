"""
WLI (White Light Imaging) Encoder Module.

This module implements the WLI image encoder using Transformer-based vision models
(Swin Transformer V2 or Vision Transformer) for extracting features from white light
endoscopy images.
"""

import torch
import torch.nn as nn
from typing import Optional
import torchvision.models as models
from torchvision.models import (
    Swin_V2_B_Weights, Swin_V2_S_Weights,
    ViT_L_16_Weights, ViT_B_16_Weights
)


class WLIEncoder(nn.Module):
    """
    WLI Image Encoder using Transformer-based vision models.

    Supports:
    - Swin Transformer V2 (Base/Small)
    - Vision Transformer (Large/Base)

    Args:
        model_name: Name of the backbone model. Options:
            - 'swin_v2_b': Swin Transformer V2 Base
            - 'swin_v2_s': Swin Transformer V2 Small
            - 'vit_l_16': Vision Transformer Large (patch size 16)
            - 'vit_b_16': Vision Transformer Base (patch size 16)
        pretrained: Whether to use pretrained weights from ImageNet.
        output_dim: Desired output feature dimension.
        freeze_layers: Number of layers to freeze for transfer learning.
    """

    SUPPORTED_MODELS = {
        'swin_v2_b': (models.swin_v2_b, Swin_V2_B_Weights.DEFAULT, 1024),
        'swin_v2_s': (models.swin_v2_s, Swin_V2_S_Weights.DEFAULT, 768),
        'vit_l_16': (models.vit_l_16, ViT_L_16_Weights.DEFAULT, 1024),
        'vit_b_16': (models.vit_b_16, ViT_B_16_Weights.DEFAULT, 768),
    }

    def __init__(
        self,
        model_name: str = 'swin_v2_b',
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
        if 'swin' in self.model_name:
            # Swin Transformer: remove the head
            self.backbone.head = nn.Identity()
        elif 'vit' in self.model_name:
            # ViT: remove the heads (classification head)
            self.backbone.heads = nn.Identity()

    def _freeze_layers(self, num_layers: int):
        """Freeze the first num_layers of the backbone."""
        if 'swin' in self.model_name:
            # Swin Transformer: freeze patch embedding and first N stages
            for param in self.backbone.features[:num_layers].parameters():
                param.requires_grad = False
        elif 'vit' in self.model_name:
            # ViT: freeze patch embedding and first N encoder blocks
            for param in self.backbone.conv_proj.parameters():
                param.requires_grad = False
            for i, block in enumerate(self.backbone.encoder.layers):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the WLI encoder.

        Args:
            x: Input WLI image tensor of shape (B, C, H, W).
               Expected: C=3 (RGB), H=W=224 (or model-specific size).

        Returns:
            Feature tensor of shape (B, output_dim).
        """
        # Extract features using backbone
        if 'swin' in self.model_name:
            # Swin Transformer forward
            features = self.backbone.features(x)
            features = self.backbone.norm(features)
            features = self.backbone.permute(features)
            features = self.backbone.avgpool(features)
            features = self.backbone.flatten(features)
        elif 'vit' in self.model_name:
            # ViT forward (get CLS token)
            x = self.backbone._process_input(x)
            n = x.shape[0]
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.backbone.encoder(x)
            features = x[:, 0]  # CLS token
        else:
            features = self.backbone(x)

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


class WLIEncoderWithAttentionPool(WLIEncoder):
    """
    WLI Encoder with attention pooling for enhanced feature aggregation.

    This variant adds an attention-based pooling mechanism to better capture
    important local features from the WLI images.
    """

    def __init__(
        self,
        model_name: str = 'swin_v2_b',
        pretrained: bool = True,
        output_dim: int = 1024,
        freeze_layers: int = 0,
        num_attention_heads: int = 8
    ):
        super().__init__(model_name, pretrained, output_dim, freeze_layers)

        # Attention pooling layer
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=self.backbone_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.backbone_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention pooling.

        Args:
            x: Input WLI image tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, output_dim).
        """
        batch_size = x.shape[0]

        if 'swin' in self.model_name:
            # Get intermediate features before final pooling
            features = self.backbone.features(x)
            features = self.backbone.norm(features)
            # features shape: (B, H', W', C) for Swin
            features = features.flatten(1, 2)  # (B, H'*W', C)
        elif 'vit' in self.model_name:
            # Get all patch tokens (excluding CLS)
            x = self.backbone._process_input(x)
            n = x.shape[0]
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            features = self.backbone.encoder(x)
            features = features[:, 1:]  # Exclude CLS token

        # Apply attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pool(query, features, features)
        pooled_features = pooled_features.squeeze(1)

        # Project to output dimension
        features = self.projection(pooled_features)

        return features


def create_wli_encoder(
    model_name: str = 'swin_v2_b',
    pretrained: bool = True,
    output_dim: int = 1024,
    freeze_layers: int = 0,
    use_attention_pool: bool = False
) -> WLIEncoder:
    """
    Factory function to create WLI encoder.

    Args:
        model_name: Backbone model name.
        pretrained: Use pretrained weights.
        output_dim: Output feature dimension.
        freeze_layers: Number of layers to freeze.
        use_attention_pool: Use attention pooling variant.

    Returns:
        WLIEncoder instance.
    """
    if use_attention_pool:
        return WLIEncoderWithAttentionPool(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            freeze_layers=freeze_layers
        )
    else:
        return WLIEncoder(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            freeze_layers=freeze_layers
        )

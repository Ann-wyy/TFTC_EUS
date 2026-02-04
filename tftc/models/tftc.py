"""
Trimodal Fusion Transformer Classifier (TFTC) Main Model.

This module implements the complete TFTC model for submucosal tumor (SMT)
classification using trimodal fusion of:
1. White Light Imaging (WLI) features
2. Endoscopic Ultrasound (EUS) features
3. Organ Location metadata

Architecture:
    WLI Image → WLI Encoder (Swin Transformer) → F_WLI
    EUS Image → EUS Encoder (ConvNeXt/ResNet) → F_EUS
    Location  → Location Embedder             → F_LOC

    [F_WLI, F_EUS, F_LOC] → Concat → Fusion Transformer → Classification Head → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .wli_encoder import WLIEncoder, create_wli_encoder
from .eus_encoder import EUSEncoder, create_eus_encoder
from .location_embedder import LocationEmbedder, create_location_embedder
from .fusion_transformer import (
    FusionTransformerEncoder,
    ClassificationHead,
    create_fusion_module,
    create_classification_head
)


class TFTC(nn.Module):
    """
    Trimodal Fusion Transformer Classifier for SMT classification.

    This model fuses features from three modalities:
    1. WLI (White Light Imaging) - surface morphology, color, vessel patterns
    2. EUS (Endoscopic Ultrasound) - deep structure, origin layer, echo features
    3. Location (Organ Position) - clinical prior knowledge

    Args:
        wli_encoder_config: Configuration dict for WLI encoder.
        eus_encoder_config: Configuration dict for EUS encoder.
        location_config: Configuration dict for location embedder.
        fusion_config: Configuration dict for fusion transformer.
        classifier_config: Configuration dict for classification head.
        active_modalities: List of active modalities for ablation studies.
            Options: ["wli", "eus", "location"]. Default uses all.
    """

    def __init__(
        self,
        wli_encoder_config: Optional[Dict] = None,
        eus_encoder_config: Optional[Dict] = None,
        location_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        active_modalities: Optional[List[str]] = None
    ):
        super().__init__()

        # Default configurations
        wli_config = wli_encoder_config or {
            'model_name': 'swin_v2_b',
            'pretrained': True,
            'output_dim': 1024,
            'freeze_layers': 0
        }

        eus_config = eus_encoder_config or {
            'model_name': 'convnext_base',
            'pretrained': True,
            'output_dim': 1024,
            'freeze_layers': 0
        }

        loc_config = location_config or {
            'num_locations': 6,
            'embedding_dim': 128,
            'hidden_dim': 64,
            'use_mlp': True
        }

        # Determine active modalities
        self.active_modalities = active_modalities or ['wli', 'eus', 'location']

        # Initialize encoders based on active modalities
        self.wli_encoder = None
        self.eus_encoder = None
        self.location_embedder = None

        self.wli_dim = 0
        self.eus_dim = 0
        self.loc_dim = 0

        if 'wli' in self.active_modalities:
            self.wli_encoder = create_wli_encoder(**wli_config)
            self.wli_dim = wli_config['output_dim']

        if 'eus' in self.active_modalities:
            self.eus_encoder = create_eus_encoder(**eus_config)
            self.eus_dim = eus_config['output_dim']

        if 'location' in self.active_modalities:
            self.location_embedder = create_location_embedder(
                num_locations=loc_config.get('num_locations', 6),
                embedding_dim=loc_config.get('embedding_dim', 128),
                hidden_dim=loc_config.get('hidden_dim', 64),
                variant='mlp' if loc_config.get('use_mlp', True) else 'linear'
            )
            self.loc_dim = loc_config.get('embedding_dim', 128)

        # Calculate fusion dimension
        self.fusion_dim = self.wli_dim + self.eus_dim + self.loc_dim

        # Fusion configuration
        fusion_cfg = fusion_config or {
            'num_layers': 3,
            'num_heads': 8,
            'ff_dim': 4096,
            'dropout': 0.1,
            'attention_dropout': 0.1
        }

        # Fusion Transformer
        self.fusion_transformer = FusionTransformerEncoder(
            hidden_dim=self.fusion_dim,
            num_layers=fusion_cfg.get('num_layers', 3),
            num_heads=fusion_cfg.get('num_heads', 8),
            ff_dim=fusion_cfg.get('ff_dim', 4096),
            dropout=fusion_cfg.get('dropout', 0.1),
            attention_dropout=fusion_cfg.get('attention_dropout', 0.1),
            use_modality_embedding=False
        )

        # Classifier configuration
        cls_config = classifier_config or {
            'hidden_dims': [512, 256],
            'num_classes': 5,
            'dropout': 0.3
        }

        # Classification Head
        self.classifier = ClassificationHead(
            input_dim=self.fusion_dim,
            hidden_dims=cls_config.get('hidden_dims', [512, 256]),
            num_classes=cls_config.get('num_classes', 5),
            dropout=cls_config.get('dropout', 0.3)
        )

    def forward(
        self,
        wli_image: Optional[torch.Tensor] = None,
        eus_image: Optional[torch.Tensor] = None,
        location: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the TFTC model.

        Args:
            wli_image: WLI image tensor of shape (B, 3, H, W).
            eus_image: EUS image tensor of shape (B, C, H, W).
            location: One-hot location tensor of shape (B, num_locations).
            return_features: If True, return intermediate features.

        Returns:
            If return_features is False:
                Logits tensor of shape (B, num_classes).
            If return_features is True:
                Tuple of (logits, feature_dict) where feature_dict contains
                intermediate features from each modality and fusion.

        Raises:
            ValueError: If required modality inputs are missing.
        """
        features = {}
        feature_list = []

        # Process WLI modality
        if 'wli' in self.active_modalities:
            if wli_image is None:
                raise ValueError("WLI image required but not provided")
            f_wli = self.wli_encoder(wli_image)
            features['wli'] = f_wli
            feature_list.append(f_wli)

        # Process EUS modality
        if 'eus' in self.active_modalities:
            if eus_image is None:
                raise ValueError("EUS image required but not provided")
            f_eus = self.eus_encoder(eus_image)
            features['eus'] = f_eus
            feature_list.append(f_eus)

        # Process Location modality
        if 'location' in self.active_modalities:
            if location is None:
                raise ValueError("Location required but not provided")
            f_loc = self.location_embedder(location)
            features['location'] = f_loc
            feature_list.append(f_loc)

        # Concatenate all features
        if len(feature_list) == 0:
            raise ValueError("At least one modality must be active")

        f_fused = torch.cat(feature_list, dim=-1)
        features['fused_pre_transformer'] = f_fused

        # Apply fusion transformer
        f_context = self.fusion_transformer(f_fused)
        features['fused_post_transformer'] = f_context

        # Classification
        logits = self.classifier(f_context)

        if return_features:
            return logits, features
        return logits

    def get_attention_weights(
        self,
        wli_image: Optional[torch.Tensor] = None,
        eus_image: Optional[torch.Tensor] = None,
        location: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from the fusion transformer for interpretability.

        This method is useful for understanding how the model weighs
        different modalities in making predictions.

        Returns:
            Dictionary containing attention weights from each transformer layer.
        """
        # This is a placeholder - actual implementation would require
        # modifying the transformer to return attention weights
        pass

    @property
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoders(self):
        """Freeze all encoder parameters for transfer learning."""
        if self.wli_encoder is not None:
            for param in self.wli_encoder.parameters():
                param.requires_grad = False
        if self.eus_encoder is not None:
            for param in self.eus_encoder.parameters():
                param.requires_grad = False
        if self.location_embedder is not None:
            for param in self.location_embedder.parameters():
                param.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True


class TFTCAblation(TFTC):
    """
    TFTC model variant for ablation studies.

    Allows easy configuration of which modalities to use for
    systematic ablation experiments.
    """

    @classmethod
    def wli_only(
        cls,
        wli_encoder_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None
    ) -> 'TFTCAblation':
        """Create model using only WLI modality."""
        return cls(
            wli_encoder_config=wli_encoder_config,
            classifier_config=classifier_config,
            active_modalities=['wli']
        )

    @classmethod
    def eus_only(
        cls,
        eus_encoder_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None
    ) -> 'TFTCAblation':
        """Create model using only EUS modality."""
        return cls(
            eus_encoder_config=eus_encoder_config,
            classifier_config=classifier_config,
            active_modalities=['eus']
        )

    @classmethod
    def wli_eus(
        cls,
        wli_encoder_config: Optional[Dict] = None,
        eus_encoder_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None
    ) -> 'TFTCAblation':
        """Create model using WLI and EUS (no location)."""
        return cls(
            wli_encoder_config=wli_encoder_config,
            eus_encoder_config=eus_encoder_config,
            fusion_config=fusion_config,
            classifier_config=classifier_config,
            active_modalities=['wli', 'eus']
        )

    @classmethod
    def wli_location(
        cls,
        wli_encoder_config: Optional[Dict] = None,
        location_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None
    ) -> 'TFTCAblation':
        """Create model using WLI and Location (no EUS)."""
        return cls(
            wli_encoder_config=wli_encoder_config,
            location_config=location_config,
            fusion_config=fusion_config,
            classifier_config=classifier_config,
            active_modalities=['wli', 'location']
        )

    @classmethod
    def eus_location(
        cls,
        eus_encoder_config: Optional[Dict] = None,
        location_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None
    ) -> 'TFTCAblation':
        """Create model using EUS and Location (no WLI)."""
        return cls(
            eus_encoder_config=eus_encoder_config,
            location_config=location_config,
            fusion_config=fusion_config,
            classifier_config=classifier_config,
            active_modalities=['eus', 'location']
        )


def create_tftc_model(
    config: Optional[Dict] = None,
    active_modalities: Optional[List[str]] = None
) -> TFTC:
    """
    Factory function to create TFTC model from configuration.

    Args:
        config: Full configuration dictionary.
        active_modalities: List of active modalities for ablation.

    Returns:
        TFTC model instance.
    """
    if config is None:
        return TFTC(active_modalities=active_modalities)

    return TFTC(
        wli_encoder_config=config.get('wli_encoder'),
        eus_encoder_config=config.get('eus_encoder'),
        location_config=config.get('location_embedder'),
        fusion_config=config.get('fusion_transformer'),
        classifier_config=config.get('classifier'),
        active_modalities=active_modalities or config.get('active_modalities')
    )


def create_tftc_from_dataclass(config) -> TFTC:
    """
    Create TFTC model from TFTCConfig dataclass.

    Args:
        config: TFTCConfig instance from configs/default_config.py

    Returns:
        TFTC model instance.
    """
    return TFTC(
        wli_encoder_config={
            'model_name': config.wli_encoder.model_name,
            'pretrained': config.wli_encoder.pretrained,
            'output_dim': config.wli_encoder.output_dim,
            'freeze_layers': config.wli_encoder.freeze_layers
        },
        eus_encoder_config={
            'model_name': config.eus_encoder.model_name,
            'pretrained': config.eus_encoder.pretrained,
            'output_dim': config.eus_encoder.output_dim,
            'freeze_layers': config.eus_encoder.freeze_layers
        },
        location_config={
            'num_locations': config.location_embedder.num_locations,
            'embedding_dim': config.location_embedder.embedding_dim,
            'hidden_dim': config.location_embedder.hidden_dim,
            'use_mlp': config.location_embedder.use_mlp
        },
        fusion_config={
            'num_layers': config.fusion_transformer.num_layers,
            'num_heads': config.fusion_transformer.num_heads,
            'ff_dim': config.fusion_transformer.ff_dim,
            'dropout': config.fusion_transformer.dropout,
            'attention_dropout': config.fusion_transformer.attention_dropout
        },
        classifier_config={
            'hidden_dims': config.classifier.hidden_dims,
            'num_classes': config.classifier.num_classes,
            'dropout': config.classifier.dropout
        }
    )

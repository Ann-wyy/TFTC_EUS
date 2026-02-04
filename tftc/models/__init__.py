"""
TFTC Models Module.

This module exports all model components for the Trimodal Fusion
Transformer Classifier.
"""

from .wli_encoder import (
    WLIEncoder,
    WLIEncoderWithAttentionPool,
    create_wli_encoder
)

from .eus_encoder import (
    EUSEncoder,
    EUSEncoderWithSPP,
    EUSEncoderWithSEBlock,
    create_eus_encoder
)

from .location_embedder import (
    LocationEmbedder,
    LearnableLocationEmbedding,
    HierarchicalLocationEmbedder,
    LocationEmbedderWithPrior,
    create_location_embedder
)

from .fusion_transformer import (
    FusionTransformerEncoder,
    FusionTransformerWithCrossAttention,
    ClassificationHead,
    ClassificationHeadWithUncertainty,
    create_fusion_module,
    create_classification_head
)

from .tftc import (
    TFTC,
    TFTCAblation,
    create_tftc_model,
    create_tftc_from_dataclass
)

__all__ = [
    # WLI Encoders
    'WLIEncoder',
    'WLIEncoderWithAttentionPool',
    'create_wli_encoder',

    # EUS Encoders
    'EUSEncoder',
    'EUSEncoderWithSPP',
    'EUSEncoderWithSEBlock',
    'create_eus_encoder',

    # Location Embedders
    'LocationEmbedder',
    'LearnableLocationEmbedding',
    'HierarchicalLocationEmbedder',
    'LocationEmbedderWithPrior',
    'create_location_embedder',

    # Fusion and Classification
    'FusionTransformerEncoder',
    'FusionTransformerWithCrossAttention',
    'ClassificationHead',
    'ClassificationHeadWithUncertainty',
    'create_fusion_module',
    'create_classification_head',

    # Main Model
    'TFTC',
    'TFTCAblation',
    'create_tftc_model',
    'create_tftc_from_dataclass',
]

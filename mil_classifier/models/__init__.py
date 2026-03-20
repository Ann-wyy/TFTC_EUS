"""
模型模块

导出所有模型组件
"""

from .relational_attention import RelationalAttention

from .encoders import (
    BaseEncoder,
    UltrasoundEncoder,
    WhiteLightEncoder,
    create_encoder
)

from .fusion import (
    ConcatFusion,
    CrossAttentionFusion,
    MultiModalFusion
)

from .mil_pooling import (
    AttentionMILPooling,
    GatedAttentionMILPooling,
    TransformerMILPooling,
    MaxMILPooling,
    MeanMILPooling,
    MILPooling
)

from .classifier import (
    PatientClassifier,
    FrameClassifier,
    MultiTaskClassifier,
    ClassificationHead
)

from .mil_model import (
    MultiModalMILClassifier,
)

__all__ = [
    # Relational Attention (from LA-RANet)
    'RelationalAttention',

    # Encoders
    'BaseEncoder',
    'UltrasoundEncoder',
    'WhiteLightEncoder',
    'create_encoder',

    # Fusion
    'ConcatFusion',
    'CrossAttentionFusion',
    'MultiModalFusion',

    # MIL Pooling
    'AttentionMILPooling',
    'GatedAttentionMILPooling',
    'TransformerMILPooling',
    'MaxMILPooling',
    'MeanMILPooling',
    'MILPooling',

    # Classifier
    'PatientClassifier',
    'FrameClassifier',
    'MultiTaskClassifier',
    'ClassificationHead',

    # Main Model
    'MultiModalMILClassifier',
]

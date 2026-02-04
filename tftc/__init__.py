"""
TFTC - Trimodal Fusion Transformer Classifier

A deep learning model for submucosal tumor (SMT) classification using
trimodal fusion of:
1. White Light Imaging (WLI) - Swin Transformer based encoder
2. Endoscopic Ultrasound (EUS) - ConvNeXt/ResNet based encoder
3. Location Metadata - MLP based embedder

The model fuses features from all three modalities using a Transformer
encoder and performs multi-class classification of tumor types.
"""

__version__ = '1.0.0'
__author__ = 'TFTC Team'

from .models import (
    TFTC,
    TFTCAblation,
    create_tftc_model,
    create_tftc_from_dataclass
)

__all__ = [
    'TFTC',
    'TFTCAblation',
    'create_tftc_model',
    'create_tftc_from_dataclass',
]

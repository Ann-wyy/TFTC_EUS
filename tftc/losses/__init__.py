"""
TFTC Losses Module.

This module exports loss functions for training.
"""

from .focal_loss import (
    FocalLoss,
    WeightedCrossEntropyLoss,
    LabelSmoothingLoss,
    CombinedLoss,
    create_loss_function
)

__all__ = [
    'FocalLoss',
    'WeightedCrossEntropyLoss',
    'LabelSmoothingLoss',
    'CombinedLoss',
    'create_loss_function',
]

"""
损失函数模块
"""

from .losses import (
    FocalLoss,
    ClassBalancedLoss,
    MILLoss,
    create_loss_function
)

__all__ = [
    'FocalLoss',
    'ClassBalancedLoss',
    'MILLoss',
    'create_loss_function',
]

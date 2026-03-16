"""
精简版损失模块 - GPU/CPU 无忧版

功能:
- Focal Loss (硬标签/软标签)
- Class-Balanced Loss
- MIL Loss (病人级 + 可选帧级辅助)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

# ------------------------
# 通用 Focal Loss 核心
# ------------------------
def focal_loss_core(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    通用 Focal Loss, 支持硬标签
    """
    p = F.softmax(inputs, dim=-1)
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1 - p_t) ** gamma
    if alpha is not None:
        # <<< 确保 alpha 在和 targets 同设备 >>>
        alpha = alpha.to(targets.device)
        alpha_t = alpha.gather(0, targets)
        focal_weight = alpha_t * focal_weight
    loss = focal_weight * ce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# ------------------------
# Focal Loss 模块
# ------------------------
class FocalLoss(nn.Module):
    """支持硬标签和软标签的 Focal Loss"""
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 2:
            # 软标签
            p = F.softmax(inputs, dim=-1)
            focal_weight = (1 - p) ** self.gamma
            log_p = F.log_softmax(inputs, dim=-1)
            loss = -(targets * focal_weight * log_p)
            if self.alpha is not None:
                alpha = self.alpha.to(inputs.device)
                # 如果软标签是概率形式，alpha 广播
                loss = loss * alpha.unsqueeze(0)
            loss = loss.sum(dim=-1)
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss
        else:
            # 硬标签
            return focal_loss_core(inputs, targets, gamma=self.gamma, alpha=self.alpha, reduction=self.reduction)

# ------------------------
# Class-Balanced Loss
# ------------------------
class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class: List[int], beta: float = 0.9999, gamma: float = 2.0, loss_type: str = 'focal'):
        super().__init__()
        self.gamma = gamma
        self.loss_type = loss_type

        effective_num = 1 - torch.pow(beta, torch.tensor(samples_per_class, dtype=torch.float))
        weights = (1 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)
        self.register_buffer('weights', weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 确保权重在和 inputs 同一 device
        weights = self.weights.to(inputs.device)
        if self.loss_type == 'focal':
            return focal_loss_core(inputs, targets, gamma=self.gamma, alpha=weights)
        else:
            return F.cross_entropy(inputs, targets, weight=weights)

# ------------------------
# MIL 总损失
# ------------------------
class MILLoss(nn.Module):
    """
    MIL Loss = patient_loss + λ * frame_loss
    """
    def __init__(self, patient_loss: nn.Module, auxiliary_weight: float = 0.3, use_auxiliary: bool = True):
        super().__init__()
        self.patient_loss = patient_loss
        self.auxiliary_weight = auxiliary_weight
        self.frame_loss = nn.BCEWithLogitsLoss(reduction='none') if use_auxiliary else None

    def forward(
        self,
        patient_logits: torch.Tensor,
        patient_labels: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:

        losses = {}
        # 病人级损失
        losses['patient'] = self.patient_loss(patient_logits, patient_labels)

        # 帧级辅助损失
        if self.frame_loss is not None and frame_logits is not None and frame_labels is not None:
            valid_mask = (frame_labels >= 0)
            if mask is not None:
                valid_mask = valid_mask & mask
            if valid_mask.any():
                frame_loss = self.frame_loss(frame_logits[valid_mask].float(), frame_labels[valid_mask].float())
                losses['frame'] = frame_loss.mean()
            else:
                losses['frame'] = torch.tensor(0.0, device=patient_logits.device)
        else:
            losses['frame'] = torch.tensor(0.0, device=patient_logits.device)

        # 总损失
        losses['total'] = losses['patient'] + self.auxiliary_weight * losses['frame']
        return losses

# ------------------------
# 创建统一损失函数
# ------------------------
def create_loss_function(
    loss_type: str = 'focal',
    num_classes: int = 6,
    class_weights: Optional[List[float]] = None,
    samples_per_class: Optional[List[int]] = None,
    gamma: float = 2.0,
    auxiliary_weight: float = 0.3,
    use_auxiliary: bool = True
) -> MILLoss:

    if loss_type == 'focal':
        alpha = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        patient_loss = FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'class_balanced':
        if samples_per_class is None:
            samples_per_class = [100] * num_classes
        patient_loss = ClassBalancedLoss(samples_per_class, gamma=gamma)
    elif loss_type == 'ce':
        weight = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        patient_loss = nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"未知 loss_type: {loss_type}")

    return MILLoss(patient_loss=patient_loss, auxiliary_weight=auxiliary_weight, use_auxiliary=use_auxiliary)

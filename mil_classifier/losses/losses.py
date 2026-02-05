"""
损失函数模块

包含:
- Focal Loss: 处理类别不平衡
- Class-Balanced Loss: 基于有效样本数的加权
- MIL Loss: 结合病人级和帧级损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: 类别权重
        gamma: 聚焦参数 (默认2.0)
        reduction: 'none', 'mean', 'sum'
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: logits [B, C]
            targets: 标签 [B] 或软标签 [B, C]

        Returns:
            损失值
        """
        # 处理软标签
        if targets.dim() == 2:
            return self._forward_soft_labels(inputs, targets)

        # 计算概率
        p = F.softmax(inputs, dim=-1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 获取目标类别的概率
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # 应用alpha
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def _forward_soft_labels(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """处理软标签"""
        p = F.softmax(inputs, dim=-1)
        focal_weight = (1 - p) ** self.gamma

        log_p = F.log_softmax(inputs, dim=-1)
        loss = -targets * focal_weight * log_p

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            loss = loss * alpha.unsqueeze(0)

        loss = loss.sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss

    基于有效样本数的加权损失
    权重 = (1 - β) / (1 - β^n_y)

    Args:
        samples_per_class: 每个类别的样本数
        beta: 平衡参数 (默认0.9999)
        loss_type: 'focal' 或 'ce'
    """

    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        loss_type: str = 'focal'
    ):
        super().__init__()

        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

        # 计算有效样本数权重
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class, dtype=torch.float))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)

        self.register_buffer('weights', weights)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: logits [B, C]
            targets: 标签 [B]

        Returns:
            损失值
        """
        if self.loss_type == 'focal':
            return self._focal_loss(inputs, targets)
        else:
            return F.cross_entropy(inputs, targets, weight=self.weights)

    def _focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """带权重的Focal Loss"""
        p = F.softmax(inputs, dim=-1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        # 应用类别权重
        weight_t = self.weights.gather(0, targets)
        loss = weight_t * focal_weight * ce_loss

        return loss.mean()


class MILLoss(nn.Module):
    """
    MIL损失函数

    结合病人级分类损失和帧级辅助损失

    L_total = L_patient + λ * L_frame
    """

    def __init__(
        self,
        patient_loss: nn.Module,
        auxiliary_weight: float = 0.3,
        use_auxiliary: bool = True
    ):
        super().__init__()

        self.patient_loss = patient_loss
        self.auxiliary_weight = auxiliary_weight
        self.use_auxiliary = use_auxiliary

        if use_auxiliary:
            self.frame_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        patient_logits: torch.Tensor,
        patient_labels: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            patient_logits: 病人分类logits [B, C]
            patient_labels: 病人标签 [B] 或软标签 [B, C]
            frame_logits: 帧级检测logits [B, N]
            frame_labels: 帧级标签 [B, N]
            mask: 有效帧掩码 [B, N]

        Returns:
            损失字典 {'total', 'patient', 'frame'}
        """
        losses = {}

        # 病人级损失
        patient_loss = self.patient_loss(patient_logits, patient_labels)
        losses['patient'] = patient_loss

        # 帧级辅助损失
        if self.use_auxiliary and frame_logits is not None and frame_labels is not None:
            # 过滤无标签的帧 (frame_labels == -1)
            valid_mask = (frame_labels >= 0)
            if mask is not None:
                valid_mask = valid_mask & mask

            if valid_mask.any():
                frame_loss = self.frame_loss(
                    frame_logits[valid_mask].float(),
                    frame_labels[valid_mask].float()
                )
                frame_loss = frame_loss.mean()
            else:
                frame_loss = torch.tensor(0.0, device=patient_logits.device)

            losses['frame'] = frame_loss
        else:
            losses['frame'] = torch.tensor(0.0, device=patient_logits.device)

        # 总损失
        losses['total'] = losses['patient'] + self.auxiliary_weight * losses['frame']

        return losses


def create_loss_function(
    loss_type: str = 'focal',
    num_classes: int = 6,
    class_weights: Optional[List[float]] = None,
    samples_per_class: Optional[List[int]] = None,
    gamma: float = 2.0,
    auxiliary_weight: float = 0.3,
    use_auxiliary: bool = True
) -> MILLoss:
    """
    创建损失函数

    Args:
        loss_type: 'focal', 'ce', 'class_balanced'
        num_classes: 类别数
        class_weights: 类别权重
        samples_per_class: 每个类别的样本数 (用于class_balanced)
        gamma: Focal Loss gamma参数
        auxiliary_weight: 辅助任务权重
        use_auxiliary: 是否使用辅助任务

    Returns:
        MILLoss实例
    """
    if loss_type == 'focal':
        if class_weights is not None:
            alpha = torch.tensor(class_weights, dtype=torch.float)
        else:
            alpha = None
        patient_loss = FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == 'class_balanced':
        if samples_per_class is None:
            # 默认均匀分布
            samples_per_class = [100] * num_classes
        patient_loss = ClassBalancedLoss(
            samples_per_class=samples_per_class,
            gamma=gamma
        )

    elif loss_type == 'ce':
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float)
        else:
            weight = None
        patient_loss = nn.CrossEntropyLoss(weight=weight)

    else:
        raise ValueError(f"未知的loss_type: {loss_type}")

    return MILLoss(
        patient_loss=patient_loss,
        auxiliary_weight=auxiliary_weight,
        use_auxiliary=use_auxiliary
    )

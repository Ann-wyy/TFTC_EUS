"""
分类器模块

包含:
1. 病人级分类器: 6分类 (平滑肌瘤, 脂肪瘤, 间质瘤, 神经内分泌瘤, 异位胰腺, 其他)
2. 帧级辅助分类器: 肿瘤检测 (可选)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class PatientClassifier(nn.Module):
    """
    病人级分类器

    输入: 病人级特征向量
    输出: 6分类概率
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [256],
        num_classes: int = 6,
        dropout: float = 0.5
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 病人级特征 [B, D]

        Returns:
            分类logits [B, num_classes]
        """
        return self.classifier(x)


class FrameClassifier(nn.Module):
    """
    帧级辅助分类器

    用于帧级肿瘤检测任务，增强MIL注意力学习
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 二分类: 是否包含肿瘤
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 帧级特征 [B, N, D] 或 [B*N, D]

        Returns:
            帧级肿瘤检测logits [B, N] 或 [B*N]
        """
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
            logits = self.classifier(x).squeeze(-1)
            return logits.reshape(B, N)
        else:
            return self.classifier(x).squeeze(-1)


class MultiTaskClassifier(nn.Module):
    """
    多任务分类器

    包含主分类任务和辅助任务
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [256],
        num_classes: int = 6,
        dropout: float = 0.5,
        use_auxiliary: bool = True,
        auxiliary_hidden_dim: int = 256
    ):
        super().__init__()

        self.use_auxiliary = use_auxiliary

        # 主分类器: 病人级
        self.patient_classifier = PatientClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )

        # 辅助分类器: 帧级
        if use_auxiliary:
            self.frame_classifier = FrameClassifier(
                input_dim=input_dim,
                hidden_dim=auxiliary_hidden_dim,
                dropout=dropout
            )

    def forward(
        self,
        patient_features: torch.Tensor,
        frame_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            patient_features: 病人级特征 [B, D]
            frame_features: 帧级特征 [B, N, D] (仅辅助任务需要)

        Returns:
            字典包含:
            - patient_logits: 病人分类logits [B, num_classes]
            - frame_logits: 帧级检测logits [B, N] (可选)
        """
        outputs = {}

        # 主分类
        outputs['patient_logits'] = self.patient_classifier(patient_features)

        # 辅助分类
        if self.use_auxiliary and frame_features is not None:
            outputs['frame_logits'] = self.frame_classifier(frame_features)

        return outputs


class ClassificationHead(nn.Module):
    """
    完整的分类头

    整合了病人级分类、帧级辅助任务和预测输出
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [256],
        num_classes: int = 6,
        class_names: Optional[List[str]] = None,
        dropout: float = 0.5,
        use_auxiliary: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.class_names = class_names or [
            "平滑肌瘤", "脂肪瘤", "间质瘤",
            "神经内分泌瘤", "异位胰腺", "其他"
        ]

        self.classifier = MultiTaskClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            use_auxiliary=use_auxiliary
        )

    def forward(
        self,
        patient_features: torch.Tensor,
        frame_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            patient_features: 病人级特征 [B, D]
            frame_features: 帧级特征 [B, N, D]

        Returns:
            分类输出字典
        """
        return self.classifier(patient_features, frame_features)

    def predict(
        self,
        patient_features: torch.Tensor,
        return_probs: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        预测

        Args:
            patient_features: 病人级特征 [B, D]
            return_probs: 是否返回概率

        Returns:
            预测结果字典
        """
        outputs = self.classifier(patient_features)

        logits = outputs['patient_logits']
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        results = {
            'predictions': preds,
            'logits': logits
        }

        if return_probs:
            results['probabilities'] = probs

        return results

    def get_class_name(self, class_idx: int) -> str:
        """获取类别名称"""
        return self.class_names[class_idx]

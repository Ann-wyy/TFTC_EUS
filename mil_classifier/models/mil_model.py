"""
多模态MIL肿瘤分类模型 (Multi-Modal MIL Tumor Classifier)

完整的端到端模型，整合:
1. 超声编码器
2. 白光编码器
3. 多模态融合
4. MIL注意力池化
5. 病人级分类器 (含辅助任务)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .encoders import UltrasoundEncoder, WhiteLightEncoder
from .fusion import MultiModalFusion
from .mil_pooling import MILPooling
from .classifier import ClassificationHead


class MultiModalMILClassifier(nn.Module):
    """
    多模态MIL肿瘤分类模型

    数据流:
    超声帧 [B, N, C, H, W] → 超声编码器 → h_us [B, N, D]
    白光帧 [B, N, C, H, W] → 白光编码器 → h_wli [B, N, D]
                     ↓
    多模态融合 (concat / cross-attention) → h_fused [B, N, D]
                     ↓
    MIL注意力池化 → H_patient [B, D], attention_weights [B, N]
                     ↓
    病人级分类 → 6分类概率
    (可选) 帧级辅助任务 → 帧级肿瘤检测
    """

    def __init__(
        self,
        # 编码器配置
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 512,
        freeze_early_layers: bool = True,
        freeze_until_layer: int = 2,
        # 融合配置
        fusion_type: str = 'cross_attention',
        fusion_num_heads: int = 8,
        fusion_num_layers: int = 2,
        fusion_dropout: float = 0.1,
        # MIL配置
        mil_pooling_type: str = 'gated_attention',
        mil_hidden_dim: int = 256,
        mil_dropout: float = 0.1,
        # 分类器配置
        num_classes: int = 6,
        classifier_hidden_dims: List[int] = [256],
        classifier_dropout: float = 0.5,
        use_auxiliary_task: bool = True,
        # 其他
        class_names: Optional[List[str]] = None
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_auxiliary_task = use_auxiliary_task
        self.class_names = class_names or [
            "平滑肌瘤", "脂肪瘤", "间质瘤",
            "神经内分泌瘤", "异位胰腺", "其他"
        ]

        # 1. 超声编码器
        self.ultrasound_encoder = UltrasoundEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            freeze_early_layers=freeze_early_layers,
            freeze_until_layer=freeze_until_layer
        )

        # 2. 白光编码器
        self.white_light_encoder = WhiteLightEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            freeze_early_layers=freeze_early_layers,
            freeze_until_layer=freeze_until_layer
        )

        # 3. 多模态融合
        self.fusion = MultiModalFusion(
            input_dim=feature_dim,
            output_dim=feature_dim,
            fusion_type=fusion_type,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            dropout=fusion_dropout
        )

        # 4. MIL注意力池化
        self.mil_pooling = MILPooling(
            pooling_type=mil_pooling_type,
            input_dim=feature_dim,
            hidden_dim=mil_hidden_dim,
            dropout=mil_dropout
        )

        # 5. 分类头
        self.classifier = ClassificationHead(
            input_dim=feature_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            class_names=self.class_names,
            dropout=classifier_dropout,
            use_auxiliary=use_auxiliary_task
        )

    def encode_frames(
        self,
        us_frames: torch.Tensor,
        wli_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码所有帧

        Args:
            us_frames: 超声帧 [B, N, C, H, W]
            wli_frames: 白光帧 [B, N, C, H, W]

        Returns:
            h_us: 超声特征 [B, N, D]
            h_wli: 白光特征 [B, N, D]
        """
        B, N, C, H, W = us_frames.shape

        # 展平batch和帧维度
        us_flat = us_frames.view(B * N, C, H, W)
        wli_flat = wli_frames.view(B * N, -1, H, W)

        # 编码
        h_us = self.ultrasound_encoder(us_flat)  # [B*N, D]
        h_wli = self.white_light_encoder(wli_flat)  # [B*N, D]

        # 恢复形状
        h_us = h_us.view(B, N, -1)  # [B, N, D]
        h_wli = h_wli.view(B, N, -1)  # [B, N, D]

        return h_us, h_wli

    def forward(
        self,
        us_frames: torch.Tensor,
        wli_frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            us_frames: 超声帧 [B, N, C, H, W]
            wli_frames: 白光帧 [B, N, C, H, W]
            mask: 有效帧掩码 [B, N]
            return_attention: 是否返回注意力权重
            return_features: 是否返回中间特征

        Returns:
            输出字典:
            - patient_logits: 病人分类logits [B, num_classes]
            - frame_logits: 帧级检测logits [B, N] (如果use_auxiliary_task)
            - attention_weights: MIL注意力权重 [B, N]
            - (可选) features: 中间特征字典
        """
        outputs = {}

        # 1. 编码帧
        h_us, h_wli = self.encode_frames(us_frames, wli_frames)

        if return_features:
            outputs['h_us'] = h_us
            outputs['h_wli'] = h_wli

        # 2. 多模态融合
        h_fused, fusion_attention = self.fusion(
            h_us, h_wli, return_attention
        )

        if return_features:
            outputs['h_fused'] = h_fused
        if fusion_attention is not None:
            outputs['fusion_attention'] = fusion_attention

        # 3. MIL池化
        H_patient, mil_attention = self.mil_pooling(
            h_fused, mask, return_attention
        )

        if return_features:
            outputs['H_patient'] = H_patient
        if return_attention and mil_attention is not None:
            outputs['attention_weights'] = mil_attention

        # 4. 分类
        frame_features = h_fused if self.use_auxiliary_task else None
        classifier_outputs = self.classifier(H_patient, frame_features)

        outputs['patient_logits'] = classifier_outputs['patient_logits']
        if 'frame_logits' in classifier_outputs:
            outputs['frame_logits'] = classifier_outputs['frame_logits']

        return outputs

    def predict(
        self,
        us_frames: torch.Tensor,
        wli_frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        预测接口

        Args:
            us_frames: 超声帧 [B, N, C, H, W]
            wli_frames: 白光帧 [B, N, C, H, W]
            mask: 有效帧掩码 [B, N]

        Returns:
            预测结果:
            - predictions: 预测类别 [B]
            - probabilities: 类别概率 [B, num_classes]
            - attention_weights: 注意力权重 [B, N]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                us_frames, wli_frames, mask,
                return_attention=True,
                return_features=False
            )

            logits = outputs['patient_logits']
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            return {
                'predictions': preds,
                'probabilities': probs,
                'attention_weights': outputs.get('attention_weights'),
                'class_names': [self.class_names[p.item()] for p in preds]
            }

    def get_attention_visualization(
        self,
        us_frames: torch.Tensor,
        wli_frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> Dict:
        """
        获取用于可视化的注意力信息

        Args:
            us_frames: 超声帧
            wli_frames: 白光帧
            mask: 有效帧掩码
            top_k: 返回注意力最高的前k帧

        Returns:
            可视化信息字典
        """
        outputs = self.forward(
            us_frames, wli_frames, mask,
            return_attention=True,
            return_features=True
        )

        attention = outputs.get('attention_weights')  # [B, N]
        if attention is None:
            return {}

        B, N = attention.shape

        # 获取top-k帧索引
        top_values, top_indices = attention.topk(
            min(top_k, N), dim=-1
        )

        return {
            'attention_weights': attention,
            'top_frame_indices': top_indices,
            'top_frame_weights': top_values,
            'predictions': outputs['patient_logits'].argmax(dim=-1),
            'probabilities': F.softmax(outputs['patient_logits'], dim=-1)
        }

    @property
    def num_parameters(self) -> int:
        """模型参数总数"""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """可训练参数数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config) -> MultiModalMILClassifier:
    """
    从配置创建模型

    Args:
        config: Config对象

    Returns:
        模型实例
    """
    return MultiModalMILClassifier(
        # 编码器
        backbone=config.encoder.backbone,
        pretrained=config.encoder.pretrained,
        feature_dim=config.encoder.feature_dim,
        freeze_early_layers=config.encoder.freeze_early_layers,
        freeze_until_layer=config.encoder.freeze_until_layer,
        # 融合
        fusion_type=config.fusion.fusion_type,
        fusion_num_heads=config.fusion.num_attention_heads,
        fusion_num_layers=config.fusion.num_fusion_layers,
        fusion_dropout=config.fusion.attention_dropout,
        # MIL
        mil_pooling_type=config.mil.pooling_type,
        mil_hidden_dim=config.mil.attention_hidden_dim,
        mil_dropout=config.mil.attention_dropout,
        # 分类器
        num_classes=config.data.num_classes,
        classifier_hidden_dims=config.classifier.hidden_dims,
        classifier_dropout=config.classifier.dropout,
        use_auxiliary_task=config.classifier.use_auxiliary_task,
        class_names=config.data.class_names
    )

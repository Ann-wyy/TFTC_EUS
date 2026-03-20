"""
多模态MIL肿瘤分类器 (Multi-Modal MIL Tumor Classifier)

用于消化道粘膜下肿瘤分类的深度学习模型

病理类别: 平滑肌瘤, 脂肪瘤, 间质瘤, 神经内分泌瘤, 异位胰腺, 其他

架构:
- 超声分支: CNN/ResNet/ViT 提取帧级特征
- 白光分支: CNN/ResNet/ViT 提取帧级特征
- 多模态融合: Concat 或 Cross-Attention
- MIL注意力池化: 聚合为病人级特征
- 病人级分类: 6分类
"""

__version__ = '1.0.0'

from .models import (
    MultiModalMILClassifier,
)

__all__ = [
    'MultiModalMILClassifier',
]

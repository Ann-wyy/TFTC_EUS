"""
配置文件 - 多模态MIL肿瘤分类模型

病理类别: 平滑肌瘤, 脂肪瘤, 间质瘤, 神经内分泌瘤, 异位胰腺, 其他
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """数据配置 (文件夹结构)

    数据目录结构:
        data_root/
            ├── patient_001/
            │   ├── ultrasound/
            │   │   └── *.jpg
            │   └── white_light/
            │       └── *.jpg (与ultrasound命名一致)
            └── ...
    """
    # 数据路径
    wli_root: str = "/data/truenas_B2/yyi/data/SMT_DATA/SMT_Frame_EUS_WL_v1"
    eus_root: str = "/data/truenas_B2/yyi/data/SMT_DATA/SMT_EUS_Procee"
    # label_file: str = "labels.txt"  # 标签文件

    # 文件夹结构
    ultrasound_folder: str = "ultrasound"
    white_light_folder: str = "white_light"

    # 数据集划分
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # MIL参数
    max_frames: int = 15 # 每个病人最大帧数

    # 图像尺寸
    img_size: Tuple[int, int] = (224, 224)

    # 超声图像通道: 灰度 + 边缘 + 深度/ROI
    eus_channels: int = 5
    # 白光图像通道: RGB
    wli_channels: int = 3

    # 病理类别
    num_classes: int = 6
    class_names: Optional[List[str]] = None

    # 归一化参数 (ImageNet)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class EncoderConfig:
    """编码器配置"""
    # 骨干网络选择: 'resnet50', 'resnet18', 'vit_small', 'convnext_tiny'
    backbone: str = "resnet50"
    pretrained: bool = True

    # 特征维度
    feature_dim: int = 512

    # 是否冻结早期层
    freeze_early_layers: bool = True
    # 冻结到哪一层 (对于ResNet: 0-4)
    freeze_until_layer: int = 2


@dataclass
class FusionConfig:
    """多模态融合配置"""
    # 融合方式: 'concat', 'cross_attention', 'both'
    fusion_type: str = "cross_attention"

    # 融合后的特征维度
    fused_dim: int = 512

    # Cross-Attention 配置
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    num_fusion_layers: int = 2


@dataclass
class MILConfig:
    """MIL (Multiple Instance Learning) 配置"""
    # MIL pooling 类型: 'attention', 'gated_attention', 'max', 'mean'
    pooling_type: str = "gated_attention"

    # Attention MIL 配置
    attention_hidden_dim: int = 256
    attention_dropout: float = 0.1

    # 是否返回注意力权重 (用于可视化)
    return_attention: bool = True


@dataclass
class ClassifierConfig:
    """分类器配置"""
    # 隐藏层维度
    hidden_dims: List[int] = field(default_factory=lambda: [256])
    dropout: float = 0.5

    # 辅助任务: 帧级肿瘤检测
    use_auxiliary_task: bool = True
    auxiliary_weight: float = 0.3  # 辅助损失权重


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    # 通用增强
    random_horizontal_flip: float = 0.5
    random_vertical_flip: float = 0.3
    random_rotation: int = 15

    # 白光增强
    wli_color_jitter: bool = True
    wli_brightness: float = 0.2
    wli_contrast: float = 0.2
    wli_saturation: float = 0.2
    wli_hue: float = 0.1

    # 超声增强
    eus_brightness_jitter: float = 0.15
    eus_noise_prob: float = 0.3
    eus_noise_std: float = 0.05
    eus_random_shift: int = 10

    # MixUp / CutMix
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本参数
    batch_size: int = 32  # 每个batch的病人数
    num_epochs: int = 100
    num_workers: int = 8

    # 优化器
    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01

    # 学习率调度
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # 损失函数
    loss_type: str = "focal"  # 'ce', 'focal', 'class_balanced'
    focal_gamma: float = 2.0

    # 类别权重 (处理不平衡)
    class_weights: Optional[List[float]] = None

    # 早停
    early_stopping_patience: int = 15

    # 混合精度
    use_amp: bool = True

    # 梯度裁剪
    max_grad_norm: float = 1.0

    # 检查点
    log_dir: str = "/data/truenas_B2/yyi/TFTC_EUS/checkpoints/logs"
    save_best_only: bool = True


@dataclass
class Config:
    """主配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    mil: MILConfig = field(default_factory=MILConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # 实验设置
    experiment_name: str = "mil_multimodal"
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        """配置后处理"""
        # 确保分类器输出维度正确
        pass


def get_config() -> Config:
    """获取默认配置"""
    return Config()

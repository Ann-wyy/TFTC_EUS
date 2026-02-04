# TFTC - Trimodal Fusion Transformer Classifier

用于消化道粘膜下肿瘤（SMT）多分类的三模态融合Transformer模型。

## 模型概述

TFTC (Trimodal Fusion Transformer Classifier) 是一个深度学习模型，融合三种模态的信息进行肿瘤分类：

1. **白光内镜图像 (WLI)** - 捕捉表面形态、颜色、血管纹理
2. **超声内镜图像 (EUS)** - 捕捉深层结构、起源层次、内部回声特征
3. **器官位置信息 (Location)** - 肿瘤所在消化道部位的临床先验知识

## 项目结构

```
TFTC_EUS/
├── configs/
│   └── default_config.py      # 默认配置文件
├── scripts/
│   └── train.py               # 训练脚本
├── tftc/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── augmentation.py    # 数据增强
│   │   └── dataset.py         # 数据集类
│   ├── losses/
│   │   ├── __init__.py
│   │   └── focal_loss.py      # 损失函数
│   ├── models/
│   │   ├── __init__.py
│   │   ├── eus_encoder.py     # EUS编码器 (ConvNeXt/ResNet)
│   │   ├── fusion_transformer.py  # 融合Transformer
│   │   ├── location_embedder.py   # 位置嵌入器
│   │   ├── tftc.py            # 主模型
│   │   └── wli_encoder.py     # WLI编码器 (Swin Transformer)
│   └── utils/
│       ├── __init__.py
│       └── metrics.py         # 评估指标
├── requirements.txt
└── README.md
```

## 安装

```bash
# 克隆仓库
git clone <repository_url>
cd TFTC_EUS

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

准备CSV格式的数据文件，包含以下列：

| patient_id | wli_path | eus_path | location | label |
|------------|----------|----------|----------|-------|
| 001 | wli/001.jpg | eus/001.jpg | gastric_body | gist |
| 002 | wli/002.jpg | eus/002.jpg | esophagus | esophageal_leiomyoma |

**支持的位置类别：**
- `esophagus` (食管)
- `gastric_cardia` (贲门)
- `gastric_fundus` (胃底)
- `gastric_body` (胃体)
- `gastric_antrum` (胃窦)
- `duodenum` (十二指肠)

**支持的肿瘤类别：**
- `esophageal_leiomyoma` (食管平滑肌瘤)
- `gastric_leiomyoma` (胃平滑肌瘤)
- `gist` (胃肠道间质瘤)
- `lipoma` (脂肪瘤)
- `other` (其他)

## 训练

### 基本训练

```bash
python scripts/train.py \
    --data_root /path/to/data \
    --train_csv train.csv \
    --val_csv val.csv \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4
```

### 使用不同编码器

```bash
# 使用 ViT 作为 WLI 编码器
python scripts/train.py \
    --wli_encoder vit_l_16 \
    --eus_encoder resnet50
```

### 消融实验

```bash
# 运行完整消融实验
python scripts/train.py --ablation

# 仅使用部分模态
python scripts/train.py --modalities wli eus  # 不使用位置信息
python scripts/train.py --modalities wli location  # 不使用EUS
```

## 模型架构

### 1. WLI 编码器 (WLI_Encoder)
- **类型**: Swin Transformer V2 (Base) 或 ViT L/16
- **输入**: RGB 图像 (224×224)
- **输出**: 特征向量 F_WLI ∈ ℝ^1024

### 2. EUS 编码器 (EUS_Encoder)
- **类型**: ConvNeXt (Base) 或 ResNet-50
- **输入**: 灰度/RGB 图像 (224×224)
- **输出**: 特征向量 F_EUS ∈ ℝ^1024

### 3. 位置嵌入器 (Location_Embedder)
- **类型**: MLP
- **输入**: One-hot 编码的位置信息
- **输出**: 位置特征 F_LOC ∈ ℝ^128

### 4. 融合Transformer (Fusion_Transformer)
- **类型**: 3层 Transformer Encoder
- **输入**: 拼接特征 [F_WLI, F_EUS, F_LOC]
- **输出**: 上下文感知特征

### 5. 分类头 (Classification_Head)
- **类型**: 两层全连接网络
- **输出**: 类别概率分布

## 训练细节

- **损失函数**: Focal Loss (γ=2.0) - 处理类别不平衡
- **优化器**: AdamW (lr=1e-4, weight_decay=0.01)
- **学习率调度**: Cosine Annealing with Warmup
- **数据增强**:
  - WLI: 随机裁剪、翻转、颜色抖动、高斯模糊
  - EUS: 随机裁剪、翻转、斑点噪声

## 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision) - macro/micro/weighted/per-class
- 召回率 (Recall)
- F1-Score
- 混淆矩阵 (Confusion Matrix)
- ROC曲线和AUC

## API 使用

```python
from tftc import TFTC, TFTCAblation

# 创建完整模型
model = TFTC()

# 创建消融模型 (仅使用 WLI + EUS)
model = TFTCAblation.wli_eus()

# 前向传播
logits = model(
    wli_image=wli_tensor,    # (B, 3, 224, 224)
    eus_image=eus_tensor,    # (B, 3, 224, 224)
    location=location_onehot  # (B, num_locations)
)

# 获取中间特征
logits, features = model(
    wli_image=wli_tensor,
    eus_image=eus_tensor,
    location=location_onehot,
    return_features=True
)
```

## 配置说明

详细配置请参考 `configs/default_config.py`，包括：

- `ImageConfig`: 图像预处理配置
- `WLIEncoderConfig`: WLI编码器配置
- `EUSEncoderConfig`: EUS编码器配置
- `LocationEmbedderConfig`: 位置嵌入器配置
- `FusionTransformerConfig`: 融合Transformer配置
- `ClassifierConfig`: 分类头配置
- `TrainingConfig`: 训练配置
- `AugmentationConfig`: 数据增强配置

## 消融实验设计

为验证各模态的贡献，建议进行以下实验：

1. **仅WLI**: 只使用白光内镜图像
2. **仅EUS**: 只使用超声内镜图像
3. **WLI + EUS**: 不使用位置信息
4. **WLI + Location**: 不使用EUS
5. **EUS + Location**: 不使用WLI
6. **完整模型**: WLI + EUS + Location

## License

MIT License

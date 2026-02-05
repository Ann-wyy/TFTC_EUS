# 多模态MIL肿瘤分类器 (Multi-Modal MIL Tumor Classifier)

用于消化道粘膜下肿瘤(SMT)分类的深度学习模型，基于Multiple Instance Learning (MIL)架构。

## 病理类别

- **平滑肌瘤** (Leiomyoma)
- **脂肪瘤** (Lipoma)
- **间质瘤** (GIST)
- **神经内分泌瘤** (NET)
- **异位胰腺** (Ectopic Pancreas)
- **其他** (Other)

## 模型架构

```
数据输入
    │
    ├── 超声帧 [B, N, 3, H, W] ──→ 超声编码器 (ResNet/ViT) ──→ h_us [B, N, D]
    │   (灰度 + 边缘 + 深度)
    │
    └── 白光帧 [B, N, 3, H, W] ──→ 白光编码器 (ResNet/ViT) ──→ h_wli [B, N, D]
        (RGB)
                                            │
                                            ▼
                              多模态融合 (Concat / Cross-Attention)
                                            │
                                            ▼
                                      h_fused [B, N, D]
                                            │
                                            ▼
                                MIL 注意力池化 (Gated Attention)
                                            │
                                            ▼
                                    H_patient [B, D]
                                    attention_weights [B, N]
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
            病人级分类器                                      帧级辅助分类器
            (6分类)                                          (肿瘤检测)
```

## 项目结构

```
TFTC_EUS/
├── configs/
│   └── config.py              # 配置文件
├── scripts/
│   └── train.py               # 训练脚本
├── mil_classifier/
│   ├── __init__.py
│   ├── models/
│   │   ├── encoders.py        # 超声/白光编码器
│   │   ├── fusion.py          # 多模态融合
│   │   ├── mil_pooling.py     # MIL注意力池化
│   │   ├── classifier.py      # 分类器
│   │   └── mil_model.py       # 完整模型
│   ├── data/
│   │   ├── dataset.py         # MIL数据集
│   │   └── augmentation.py    # 数据增强
│   ├── losses/
│   │   └── losses.py          # 损失函数
│   └── utils/
│       └── metrics.py         # 评估指标
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

CSV格式:

| patient_id | frame_id | eus_path | wli_path | label | frame_label |
|------------|----------|----------|----------|-------|-------------|
| 001 | 0 | eus/001_0.jpg | wli/001_0.jpg | 2 | 1 |
| 001 | 1 | eus/001_1.jpg | wli/001_1.jpg | 2 | 0 |
| 002 | 0 | eus/002_0.jpg | wli/002_0.jpg | 0 | 1 |

- `patient_id`: 病人ID
- `frame_id`: 帧序号
- `eus_path`: 超声图像路径
- `wli_path`: 白光图像路径
- `label`: 病人级标签 (0-5)
- `frame_label`: 帧级标签 (0/1, 可选)

## 训练

```bash
python scripts/train.py \
    --data_root /path/to/data \
    --train_csv train.csv \
    --val_csv val.csv \
    --epochs 100 \
    --batch_size 8 \
    --backbone resnet50 \
    --fusion_type cross_attention \
    --mil_pooling gated_attention
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--backbone` | 编码器骨干网络 | resnet50 |
| `--fusion_type` | 融合方式 | cross_attention |
| `--mil_pooling` | MIL池化方式 | gated_attention |
| `--batch_size` | 批次大小 (病人数) | 8 |
| `--lr` | 学习率 | 1e-4 |
| `--loss` | 损失函数 | focal |

### 支持的选项

**Backbone**: `resnet50`, `resnet18`, `convnext_tiny`, `vit_b_16`

**融合方式**: `concat`, `cross_attention`, `both`

**MIL池化**: `attention`, `gated_attention`, `transformer`, `max`, `mean`

**损失函数**: `focal`, `ce`, `class_balanced`

## 模型特性

### 1. 超声图像预处理

将灰度超声图像转换为3通道输入:
- 通道1: CLAHE增强的灰度图
- 通道2: Canny边缘检测
- 通道3: 形态学梯度 (层次增强)

### 2. 多模态融合

**Concat融合**: 简单拼接后通过MLP

**Cross-Attention融合**:
- 超声 attend to 白光 (获取表面特征)
- 白光 attend to 超声 (获取深层结构)

### 3. MIL注意力池化

**Gated Attention MIL**:
```
a_k = exp(W * (tanh(V * h_k) ⊙ sigmoid(U * h_k))) / Σ(...)
H = Σ(a_k * h_k)
```

门控机制可以更好地控制关键帧的注意力权重。

### 4. 辅助任务

帧级肿瘤检测作为辅助任务，增强MIL注意力学习。

### 5. 数据增强

**超声增强**:
- 亮度抖动
- 斑点噪声 (模拟超声特有噪声)
- 随机平移

**白光增强**:
- 颜色抖动
- 随机反光 (模拟内镜光源)
- 旋转/翻转

### 6. 类别平衡

- Focal Loss: 处理类别不平衡
- 平衡批次采样: 确保每个batch包含各类样本

## API使用

```python
from mil_classifier import MultiModalMILClassifier

# 创建模型
model = MultiModalMILClassifier(
    backbone='resnet50',
    fusion_type='cross_attention',
    mil_pooling_type='gated_attention',
    num_classes=6
)

# 前向传播
outputs = model(
    us_frames=eus_tensor,    # [B, N, 3, 224, 224]
    wli_frames=wli_tensor,   # [B, N, 3, 224, 224]
    mask=mask_tensor,        # [B, N] 可选
    return_attention=True
)

# 输出
logits = outputs['patient_logits']      # [B, 6]
attention = outputs['attention_weights'] # [B, N]

# 预测
results = model.predict(eus_tensor, wli_tensor)
# results['predictions']: 预测类别
# results['probabilities']: 类别概率
# results['attention_weights']: 注意力权重 (用于可视化关键帧)
```

## 可解释性

通过`attention_weights`可以可视化模型关注的关键帧:

```python
# 获取注意力最高的前5帧
viz_info = model.get_attention_visualization(
    eus_tensor, wli_tensor, top_k=5
)

top_indices = viz_info['top_frame_indices']  # 关键帧索引
top_weights = viz_info['top_frame_weights']  # 对应权重
```

## 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision): macro/micro/weighted/per-class
- 召回率 (Recall)
- F1-Score
- AUROC (每类)
- 混淆矩阵

## License

MIT License

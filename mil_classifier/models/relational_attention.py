"""
RelationalAttention - 移植自 LA-RANet (Zehebi29/LA-RANet)

原论文: "A Data-Efficient Visual Analytics Method for Human-Centered
Diagnostic Systems to Endoscopic Ultrasonography"

核心思路:
1. 对 backbone 输出的空间特征图，按 8 个方向做 GLCM 式邻域位移，得到 9 个位移图
2. 计算所有 C(9,2)=36 个成对组合，用分组卷积建模纹理关系
3. Channel Attention 加权后汇聚为 1 通道空间注意力 mask
4. mask * 原特征图 → 纹理增强的特征

用途: 嵌入 EUS / WLI 编码器，在 backbone 空间特征 → 全局 avgpool 之间，
      增强对 EUS 纹理模式的感知能力。
"""

import torch
import torch.nn as nn
from itertools import combinations


def get_glcm_stack(img: torch.Tensor, step: int = 1) -> torch.Tensor:
    """
    8方向 GLCM 邻域位移 + 原图，共 9 个通道

    Args:
        img: [B, 1, H, W]  (单通道均值图)
        step: 位移步长

    Returns:
        [B, 9, H, W]
    """
    gl3 = torch.cat((img[:, :, :, step:],  img[:, :, :, -step:]),  dim=3)
    gl6 = torch.cat((img[:, :, step:, :],  img[:, :, -step:, :]),  dim=2)
    gl7 = torch.cat((img[:, :, :step, :],  img[:, :, :-step, :]),  dim=2)
    gl8 = torch.cat((img[:, :, :step, :],  img[:, :, :-step, :]),  dim=2)

    trans_img = torch.cat([
        img,
        torch.cat((img[:, :, :, step:],  img[:, :, :, -step:]),  dim=3),
        torch.cat((img[:, :, step:, :],  img[:, :, -step:, :]),  dim=2),
        torch.cat((gl3[:, :, step:, :],  gl3[:, :, -step:, :]),  dim=2),
        torch.cat((img[:, :, :, :step],  img[:, :, :, :-step]), dim=3),
        torch.cat((img[:, :, :step, :],  img[:, :, :-step, :]),  dim=2),
        torch.cat((gl6[:, :, :, step:],  gl6[:, :, :, -step:]),  dim=3),
        torch.cat((gl7[:, :, :, step:],  gl7[:, :, :, -step:]),  dim=3),
        torch.cat((gl8[:, :, :, :step],  gl8[:, :, :, :-step]), dim=3),
    ], dim=1)   # [B, 9, H, W]

    return trans_img


class ChannelAttentionModule(nn.Module):
    """轻量 Channel Attention (SE-style)"""

    def __init__(self, channel: int, ratio: int = 2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.shared_MLP(self.avg_pool(x)))


class RelationalAttention(nn.Module):
    """
    GLCM 关系注意力模块

    输入: 任意通道数的空间特征图 [B, C, H, W]
    输出: 空间注意力 mask          [B, 1, H, W]  (sigmoid, 值域 [0,1])

    使用方式:
        attn = RelationalAttention()
        mask = attn(feat_map)          # [B, 1, H, W]
        feat_map = feat_map * mask     # 纹理加权
    """

    N_DIRS = 9                          # GLCM 方向数
    N_PAIRS = N_DIRS * (N_DIRS - 1) // 2   # C(9,2) = 36

    def __init__(self):
        super().__init__()
        n = self.N_PAIRS   # 36
        self._pairs = list(combinations(range(self.N_DIRS), 2))

        # 分组卷积：72 → 36，每对通道独立处理
        self.group_conv = nn.Conv2d(2 * n, n, kernel_size=1, bias=False, groups=n)
        self.bn_group   = nn.BatchNorm2d(n)
        self.relu       = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(n, n, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(n)

        self.channel_attention = ChannelAttentionModule(n)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 通道均值 → 单通道图
        avg = torch.mean(x, dim=1, keepdim=True)   # [B, 1, H, W]

        # 2. 8方向 GLCM 位移 → [B, 9, H, W]
        glcm = get_glcm_stack(avg)

        # 3. 逐对拼接 → [B, 72, H, W]
        pairs = [
            torch.cat([
                glcm[:, i, :, :].unsqueeze(1),
                glcm[:, j, :, :].unsqueeze(1),
            ], dim=1)
            for i, j in self._pairs
        ]
        feas = torch.cat(pairs, dim=1)   # [B, 72, H, W]

        # 4. 分组卷积 + BN + ReLU → [B, 36, H, W]
        feas = self.relu(self.bn_group(self.group_conv(feas)))

        # 5. 1x1 Conv + BN
        feas = self.bn1(self.conv1(feas))

        # 6. Channel Attention 加权
        feas = self.channel_attention(feas) * feas   # [B, 36, H, W]

        # 7. 通道求和 → 1 通道空间 mask
        feas = torch.sum(feas, dim=1, keepdim=True)  # [B, 1, H, W]
        return self.sigmoid(feas)

"""
多模态融合模块

支持两种融合方式:
1. Concat融合: 简单拼接后通过MLP
2. Cross-Attention融合: 使用Transformer交叉注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ConcatFusion(nn.Module):
    """
    拼接融合模块

    将超声特征和白光特征拼接后通过MLP映射
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(
        self,
        h_us: torch.Tensor,
        h_wli: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            h_us: 超声特征 [B, D] 或 [B, N, D]
            h_wli: 白光特征 [B, D] 或 [B, N, D]

        Returns:
            融合特征 [B, D] 或 [B, N, D]
        """
        # 拼接
        h_concat = torch.cat([h_us, h_wli], dim=-1)

        # 融合
        h_fused = self.fusion(h_concat)

        return h_fused


class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层

    让一个模态的特征attend到另一个模态
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value 投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            query: 查询特征 [B, N, D] 或 [B, D]
            key_value: 键值特征 [B, M, D] 或 [B, D]
            return_attention: 是否返回注意力权重

        Returns:
            输出特征, (可选) 注意力权重
        """
        # 处理2D输入
        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)

        B, N, D = query.shape
        _, M, _ = key_value.shape

        # 计算Q, K, V
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        attn_output = self.out_proj(attn_output)

        # 残差连接 + LayerNorm
        x = self.norm1(query + attn_output)

        # FFN
        x = self.norm2(x + self.ffn(x))

        if squeeze_output:
            x = x.squeeze(1)

        if return_attention:
            return x, attn_weights.mean(dim=1)  # 平均所有头的注意力
        return x, None


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块

    使用双向交叉注意力:
    - 超声 attend to 白光 (获取表面特征)
    - 白光 attend to 超声 (获取深层结构)
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: Optional[int] = None
    ):
        super().__init__()

        self.num_layers = num_layers
        output_dim = output_dim or dim

        # 多层交叉注意力
        self.us_to_wli_layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.wli_to_us_layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(
        self,
        h_us: torch.Tensor,
        h_wli: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        前向传播

        Args:
            h_us: 超声特征 [B, D] 或 [B, N, D]
            h_wli: 白光特征 [B, D] 或 [B, N, D]
            return_attention: 是否返回注意力权重

        Returns:
            融合特征, (可选) 注意力权重字典
        """
        attention_weights = {'us_to_wli': [], 'wli_to_us': []}

        # 多层交叉注意力
        for i in range(self.num_layers):
            # 超声 attend to 白光
            h_us_new, attn_us = self.us_to_wli_layers[i](
                h_us, h_wli, return_attention
            )
            # 白光 attend to 超声
            h_wli_new, attn_wli = self.wli_to_us_layers[i](
                h_wli, h_us, return_attention
            )

            h_us = h_us_new
            h_wli = h_wli_new

            if return_attention:
                attention_weights['us_to_wli'].append(attn_us)
                attention_weights['wli_to_us'].append(attn_wli)

        # 拼接并融合
        h_fused = torch.cat([h_us, h_wli], dim=-1)
        h_fused = self.final_fusion(h_fused)

        if return_attention:
            return h_fused, attention_weights
        return h_fused, None


class MultiModalFusion(nn.Module):
    """
    多模态融合模块

    支持多种融合方式
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        fusion_type: str = 'cross_attention',
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            self.fusion = ConcatFusion(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout
            )
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                dim=input_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=output_dim
            )
        elif fusion_type == 'both':
            # 同时使用两种方式
            self.concat_fusion = ConcatFusion(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout
            )
            self.cross_attention_fusion = CrossAttentionFusion(
                dim=input_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=output_dim
            )
            # 合并两种融合结果
            self.combine = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        else:
            raise ValueError(f"未知的fusion_type: {fusion_type}")

    def forward(
        self,
        h_us: torch.Tensor,
        h_wli: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        前向传播

        Args:
            h_us: 超声特征 [B, D] 或 [B, N, D]
            h_wli: 白光特征 [B, D] 或 [B, N, D]
            return_attention: 是否返回注意力权重

        Returns:
            融合特征, (可选) 注意力权重
        """
        if self.fusion_type == 'concat':
            h_fused = self.fusion(h_us, h_wli)
            return h_fused, None

        elif self.fusion_type == 'cross_attention':
            return self.fusion(h_us, h_wli, return_attention)

        elif self.fusion_type == 'both':
            h_concat = self.concat_fusion(h_us, h_wli)
            h_cross, attn = self.cross_attention_fusion(
                h_us, h_wli, return_attention
            )
            h_fused = self.combine(torch.cat([h_concat, h_cross], dim=-1))
            return h_fused, attn

        return None, None

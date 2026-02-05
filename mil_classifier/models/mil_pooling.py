"""
MIL (Multiple Instance Learning) 注意力池化模块

将帧级特征聚合为病人级特征

支持的池化方式:
1. Attention MIL: 标准注意力机制
2. Gated Attention MIL: 门控注意力机制
3. Max Pooling
4. Mean Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionMILPooling(nn.Module):
    """
    标准注意力MIL池化

    a_k = exp(W * tanh(V * h_k)) / sum(exp(W * tanh(V * h_j)))
    H = sum(a_k * h_k)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            h: 帧级特征 [B, N, D] (N为帧数)
            mask: 有效帧掩码 [B, N], True表示有效
            return_attention: 是否返回注意力权重

        Returns:
            病人级特征 [B, D]
            (可选) 注意力权重 [B, N]
        """
        B, N, D = h.shape

        # 计算注意力分数
        attn_scores = self.attention(h).squeeze(-1)  # [B, N]

        # 应用mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N]

        # 加权求和
        H = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # [B, D]

        if return_attention:
            return H, attn_weights
        return H, None


class GatedAttentionMILPooling(nn.Module):
    """
    门控注意力MIL池化

    a_k = exp(W * (tanh(V * h_k) ⊙ sigmoid(U * h_k))) / sum(...)
    H = sum(a_k * h_k)

    门控机制可以更好地控制哪些帧应该被关注
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.attention_W = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            h: 帧级特征 [B, N, D]
            mask: 有效帧掩码 [B, N]
            return_attention: 是否返回注意力权重

        Returns:
            病人级特征 [B, D]
            (可选) 注意力权重 [B, N]
        """
        B, N, D = h.shape

        # 计算门控注意力
        v = self.attention_V(h)  # [B, N, H]
        u = self.attention_U(h)  # [B, N, H]
        attn_scores = self.attention_W(v * u).squeeze(-1)  # [B, N]

        # 应用mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N]

        # 加权求和
        H = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # [B, D]

        if return_attention:
            return H, attn_weights
        return H, None


class TransformerMILPooling(nn.Module):
    """
    基于Transformer的MIL池化

    使用可学习的CLS token来聚合所有帧的信息
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            h: 帧级特征 [B, N, D]
            mask: 有效帧掩码 [B, N]
            return_attention: 是否返回注意力权重

        Returns:
            病人级特征 [B, D]
            注意力权重 (此方法不直接返回注意力权重)
        """
        B, N, D = h.shape

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)  # [B, N+1, D]

        # 更新mask (CLS token始终有效)
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=h.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            # TransformerEncoder需要padding mask
            padding_mask = ~mask
        else:
            padding_mask = None

        # Transformer编码
        h = self.transformer(h, src_key_padding_mask=padding_mask)

        # 取CLS token作为输出
        H = self.norm(h[:, 0])

        return H, None


class MaxMILPooling(nn.Module):
    """最大池化"""

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        H, indices = h.max(dim=1)
        return H, None


class MeanMILPooling(nn.Module):
    """平均池化"""

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
            lengths = mask.sum(dim=1, keepdim=True).float()
            H = h.sum(dim=1) / lengths.clamp(min=1)
        else:
            H = h.mean(dim=1)
        return H, None


class MILPooling(nn.Module):
    """
    MIL池化模块 - 统一接口
    """

    def __init__(
        self,
        pooling_type: str = 'gated_attention',
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            self.pooling = AttentionMILPooling(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif pooling_type == 'gated_attention':
            self.pooling = GatedAttentionMILPooling(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif pooling_type == 'transformer':
            self.pooling = TransformerMILPooling(
                input_dim=input_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        elif pooling_type == 'max':
            self.pooling = MaxMILPooling()
        elif pooling_type == 'mean':
            self.pooling = MeanMILPooling()
        else:
            raise ValueError(f"未知的pooling_type: {pooling_type}")

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            h: 帧级特征 [B, N, D]
            mask: 有效帧掩码 [B, N]
            return_attention: 是否返回注意力权重

        Returns:
            病人级特征 [B, D]
            (可选) 注意力权重 [B, N]
        """
        return self.pooling(h, mask, return_attention)

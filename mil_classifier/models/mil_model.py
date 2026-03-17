import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoders import UltrasoundEncoder, WhiteLightEncoder
from .fusion import CrossAttentionFusion


# --------------------------------------------------
# Positional Encoding
# --------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=256):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, dim, 2) *
            (-torch.log(torch.tensor(10000.0)) / dim)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------
# Frame Transformer
# --------------------------------------------------

class FrameTransformer(nn.Module):

    def __init__(self, dim=512, heads=8, layers=2):

        super().__init__()

        self.pos = PositionalEncoding(dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            dim_feedforward=dim * 4
        )

        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, x, mask=None):

        x = self.pos(x)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()

        return self.encoder(x, src_key_padding_mask=key_padding_mask)


# --------------------------------------------------
# CLAM MIL Pooling (Gated Attention + TopK)
# --------------------------------------------------

class CLAMMIL(nn.Module):

    def __init__(self, dim=512, hidden=256, k=5, num_classes=6):

        super().__init__()

        self.k = k

        # gated attention
        self.attn_V = nn.Linear(dim, hidden)
        self.attn_U = nn.Linear(dim, hidden)
        self.attn_w = nn.Linear(hidden, 1)

        # instance classifier
        self.instance_classifier = nn.Linear(dim, num_classes)

    def forward(self, x, mask=None):

        # x: B N D

        A_V = torch.tanh(self.attn_V(x))
        A_U = torch.sigmoid(self.attn_U(x))

        A = self.attn_w(A_V * A_U).squeeze(-1)

        if mask is not None:
            fill_value = torch.finfo(A.dtype).min  # FP16 时约 -65504
            A = A.masked_fill(~mask, fill_value)

        A = torch.softmax(A, dim=1)

        # bag feature
        bag_feat = torch.sum(A.unsqueeze(-1) * x, dim=1)

        # TopK instances
        topk = torch.topk(A, min(self.k, x.size(1)), dim=1).indices

        inst_feat = torch.gather(
            x,
            1,
            topk.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )

        inst_logits = self.instance_classifier(inst_feat)

        return bag_feat, A, inst_logits


# --------------------------------------------------
# SOTA MultiModal MIL Model
# --------------------------------------------------

class MultiModalMILClassifier(nn.Module):

    def __init__(
        self,
        backbone="resnet50",
        feature_dim=512,
        num_classes=6,
        pretrained=True,
        k=5,
        eus_channels=3,
    ):

        super().__init__()

        # encoders
        self.eus_encoder = UltrasoundEncoder(
            backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            input_channels=eus_channels,
        )

        self.wli_encoder = WhiteLightEncoder(
            backbone,
            pretrained=pretrained,
            feature_dim=feature_dim
        )

        # frame modeling
        self.eus_transformer = FrameTransformer(feature_dim)
        self.wli_transformer = FrameTransformer(feature_dim)

        # cross modal fusion — 双向 CrossAttentionFusion
        self.cross_modal = CrossAttentionFusion(
            dim=feature_dim,
            num_heads=8,
            num_layers=2,
            dropout=0.1,
            output_dim=feature_dim,
        )

        # CLAM MIL
        self.mil = CLAMMIL(
            dim=feature_dim,
            hidden=256,
            k=k,
            num_classes=num_classes
        )

        # classifier
        self.classifier = nn.Linear(feature_dim, num_classes)

    # --------------------------------------------------

    def forward(
        self,
        eus_frames: torch.Tensor,
        wli_frames: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ):

        B, N, Ce, H, W = eus_frames.shape
        B, N, Cw, H, W = wli_frames.shape

        # flatten frames
        eus_frames = eus_frames.view(B * N, Ce, H, W)
        wli_frames = wli_frames.view(B * N, Cw, H, W)

        # encode
        eus_feat = self.eus_encoder(eus_frames)
        wli_feat = self.wli_encoder(wli_frames)

        eus_feat = eus_feat.view(B, N, -1)
        wli_feat = wli_feat.view(B, N, -1)

        # frame modeling
        eus_feat = self.eus_transformer(eus_feat, masks)
        wli_feat = self.wli_transformer(wli_feat, masks)

        # cross modal fusion (双向，返回 tuple: fused, attn_weights)
        fused, _ = self.cross_modal(eus_feat, wli_feat)

        # MIL pooling
        bag_feat, attn, inst_logits = self.mil(fused, masks)

        # patient classifier
        patient_logits = self.classifier(bag_feat)

        return {
            "patient_logits": patient_logits,
            "frame_logits": inst_logits,
            "attention": attn
        }

    @property
    def num_parameters(self):

        return sum(p.numel() for p in self.parameters())

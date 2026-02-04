"""
Fusion Transformer and Classification Head Module.

This module implements the multimodal feature fusion using Transformer
encoder layers and the final classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the fusion transformer.

    Adds positional information to distinguish between different modality features.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ModalityTypeEmbedding(nn.Module):
    """
    Learnable modality type embedding to distinguish between WLI, EUS, and Location.
    """

    def __init__(self, num_modalities: int = 3, d_model: int = 2176):
        super().__init__()
        self.modality_embedding = nn.Embedding(num_modalities, d_model)

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Add modality type embedding to input features.

        Args:
            x: Feature tensor of shape (B, seq_len, d_model)
            modality_ids: Modality indices of shape (seq_len,)

        Returns:
            Features with modality embedding added.
        """
        modality_emb = self.modality_embedding(modality_ids)
        return x + modality_emb.unsqueeze(0)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer for explicit modality interaction.

    Allows each modality to attend to features from other modalities.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-modal attention.

        Args:
            query: Query tensor from one modality (B, 1, d_model)
            key_value: Key/Value tensor from other modalities (B, N, d_model)

        Returns:
            Attended features (B, 1, d_model)
        """
        attended, _ = self.attention(query, key_value, key_value)
        return self.norm(query + self.dropout(attended))


class FusionTransformerEncoder(nn.Module):
    """
    Fusion Transformer Encoder for multimodal feature integration.

    Uses stacked Transformer encoder layers with multi-head self-attention
    to capture cross-modal relationships and dependencies.

    Args:
        hidden_dim: Input/output feature dimension.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward network dimension.
        dropout: Dropout rate.
        attention_dropout: Attention dropout rate.
        use_modality_embedding: Whether to add modality type embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 2176,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 4096,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_modality_embedding: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_modality_embedding = use_modality_embedding

        # Modality type embedding
        if use_modality_embedding:
            self.modality_embedding = ModalityTypeEmbedding(
                num_modalities=3,
                d_model=hidden_dim
            )

        # Input layer normalization
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer normalization
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through fusion transformer.

        Args:
            x: Fused feature tensor of shape (B, hidden_dim) or (B, seq_len, hidden_dim).
            modality_ids: Optional modality indices for embedding.
            attention_mask: Optional attention mask.

        Returns:
            Context-aware fused features of shape (B, hidden_dim) or (B, seq_len, hidden_dim).
        """
        # Handle 2D input (single fused vector)
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, hidden_dim)
            squeeze_output = True

        # Input normalization
        x = self.input_norm(x)

        # Add modality embedding if provided
        if self.use_modality_embedding and modality_ids is not None:
            x = self.modality_embedding(x, modality_ids)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)

        # Output normalization
        x = self.output_norm(x)

        # Squeeze back to 2D if input was 2D
        if squeeze_output:
            x = x.squeeze(1)

        return x


class FusionTransformerWithCrossAttention(nn.Module):
    """
    Enhanced Fusion Transformer with explicit cross-modal attention.

    This variant maintains separate representations for each modality
    and uses cross-attention for inter-modal interaction before fusion.
    """

    def __init__(
        self,
        wli_dim: int = 1024,
        eus_dim: int = 1024,
        loc_dim: int = 128,
        hidden_dim: int = 2176,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 4096,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project each modality to common dimension
        self.wli_proj = nn.Linear(wli_dim, hidden_dim)
        self.eus_proj = nn.Linear(eus_dim, hidden_dim)
        self.loc_proj = nn.Linear(loc_dim, hidden_dim)

        # Cross-modal attention layers
        self.wli_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.eus_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)

        # Fusion transformer
        self.fusion_transformer = FusionTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            use_modality_embedding=True
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(
        self,
        wli_features: torch.Tensor,
        eus_features: torch.Tensor,
        loc_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with cross-modal attention.

        Args:
            wli_features: WLI features (B, wli_dim)
            eus_features: EUS features (B, eus_dim)
            loc_features: Location features (B, loc_dim)

        Returns:
            Fused features (B, hidden_dim)
        """
        batch_size = wli_features.shape[0]

        # Project to common dimension
        wli = self.wli_proj(wli_features).unsqueeze(1)  # (B, 1, hidden_dim)
        eus = self.eus_proj(eus_features).unsqueeze(1)  # (B, 1, hidden_dim)
        loc = self.loc_proj(loc_features).unsqueeze(1)  # (B, 1, hidden_dim)

        # Cross-modal attention
        # WLI attends to EUS and Location
        wli_context = torch.cat([eus, loc], dim=1)
        wli_attended = self.wli_cross_attn(wli, wli_context)

        # EUS attends to WLI and Location
        eus_context = torch.cat([wli, loc], dim=1)
        eus_attended = self.eus_cross_attn(eus, eus_context)

        # Stack all modalities for transformer
        stacked = torch.cat([wli_attended, eus_attended, loc], dim=1)  # (B, 3, hidden_dim)

        # Create modality IDs
        modality_ids = torch.tensor([0, 1, 2], device=wli_features.device)

        # Transformer encoding
        transformed = self.fusion_transformer(stacked, modality_ids)

        # Final fusion (concatenate and reduce)
        fused = transformed.flatten(1)  # (B, 3*hidden_dim)
        output = self.final_fusion(fused)

        return output


class ClassificationHead(nn.Module):
    """
    Classification Head for final tumor type prediction.

    Uses one or more fully connected layers followed by softmax activation.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer dimensions.
        num_classes: Number of output classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2176,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 5,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

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

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.

        Args:
            x: Feature tensor of shape (B, input_dim).

        Returns:
            Logits tensor of shape (B, num_classes).
            Note: Softmax is NOT applied here for compatibility with
            CrossEntropyLoss/FocalLoss.
        """
        return self.classifier(x)


class ClassificationHeadWithUncertainty(nn.Module):
    """
    Classification Head with uncertainty estimation.

    In addition to class predictions, this head also estimates
    prediction uncertainty using Monte Carlo Dropout.
    """

    def __init__(
        self,
        input_dim: int = 2176,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 5,
        dropout: float = 0.3,
        num_mc_samples: int = 10
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_mc_samples = num_mc_samples

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

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional uncertainty estimation.

        Args:
            x: Feature tensor of shape (B, input_dim).
            return_uncertainty: Whether to compute uncertainty.

        Returns:
            Tuple of (logits, uncertainty) where uncertainty is None
            if return_uncertainty is False.
        """
        if not return_uncertainty or not self.training:
            features = self.feature_extractor(x)
            logits = self.classifier(features)
            return logits, None

        # Monte Carlo sampling for uncertainty
        self.train()  # Ensure dropout is active
        predictions = []

        for _ in range(self.num_mc_samples):
            features = self.feature_extractor(x)
            logits = self.classifier(features)
            probs = F.softmax(logits, dim=-1)
            predictions.append(probs)

        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_samples, B, num_classes)

        # Mean prediction
        mean_probs = predictions.mean(dim=0)

        # Uncertainty (entropy of mean prediction)
        uncertainty = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10),
            dim=-1
        )

        # Return logits of mean probability
        return torch.log(mean_probs + 1e-10), uncertainty


def create_fusion_module(
    wli_dim: int = 1024,
    eus_dim: int = 1024,
    loc_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 8,
    ff_dim: int = 4096,
    dropout: float = 0.1,
    variant: str = 'default'
) -> nn.Module:
    """
    Factory function to create fusion module.

    Args:
        wli_dim: WLI feature dimension.
        eus_dim: EUS feature dimension.
        loc_dim: Location feature dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward dimension.
        dropout: Dropout rate.
        variant: Module variant ('default' or 'cross_attention').

    Returns:
        Fusion module instance.
    """
    hidden_dim = wli_dim + eus_dim + loc_dim

    if variant == 'cross_attention':
        return FusionTransformerWithCrossAttention(
            wli_dim=wli_dim,
            eus_dim=eus_dim,
            loc_dim=loc_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
    else:
        return FusionTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )


def create_classification_head(
    input_dim: int = 2176,
    hidden_dims: List[int] = [512, 256],
    num_classes: int = 5,
    dropout: float = 0.3,
    with_uncertainty: bool = False
) -> nn.Module:
    """
    Factory function to create classification head.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: Hidden layer dimensions.
        num_classes: Number of classes.
        dropout: Dropout rate.
        with_uncertainty: Include uncertainty estimation.

    Returns:
        Classification head instance.
    """
    if with_uncertainty:
        return ClassificationHeadWithUncertainty(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        return ClassificationHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )

"""
Location Embedder Module.

This module implements the location information embedder for encoding
organ location metadata as a feature vector. Location information serves
as critical clinical prior knowledge for tumor classification.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class LocationEmbedder(nn.Module):
    """
    Location Information Embedder using Linear Layer or MLP.

    Encodes one-hot organ location information into a dense feature vector.

    Location categories typically include:
    - Esophagus (食管)
    - Gastric Cardia (贲门)
    - Gastric Fundus (胃底)
    - Gastric Body (胃体)
    - Gastric Antrum (胃窦)
    - Duodenum (十二指肠)

    Args:
        num_locations: Number of organ location categories.
        embedding_dim: Output embedding dimension.
        hidden_dim: Hidden layer dimension (for MLP variant).
        use_mlp: Whether to use MLP instead of simple linear layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_locations: int = 6,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        use_mlp: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_locations = num_locations
        self.embedding_dim = embedding_dim

        if use_mlp:
            self.embedder = nn.Sequential(
                nn.Linear(num_locations, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU()
            )
        else:
            self.embedder = nn.Sequential(
                nn.Linear(num_locations, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the location embedder.

        Args:
            x: One-hot encoded location tensor of shape (B, num_locations).

        Returns:
            Location embedding tensor of shape (B, embedding_dim).
        """
        return self.embedder(x)

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.embedding_dim


class LearnableLocationEmbedding(nn.Module):
    """
    Learnable Location Embedding using nn.Embedding.

    Instead of using one-hot encoding with a linear layer, this variant
    uses a learnable embedding table similar to word embeddings in NLP.

    Args:
        num_locations: Number of organ location categories.
        embedding_dim: Output embedding dimension.
    """

    def __init__(
        self,
        num_locations: int = 6,
        embedding_dim: int = 128
    ):
        super().__init__()

        self.num_locations = num_locations
        self.embedding_dim = embedding_dim

        # Learnable embedding table
        self.embedding = nn.Embedding(num_locations, embedding_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the learnable embedding.

        Args:
            x: Location indices tensor of shape (B,) with integer values
               in range [0, num_locations-1].
               OR one-hot encoded tensor of shape (B, num_locations).

        Returns:
            Location embedding tensor of shape (B, embedding_dim).
        """
        # Handle one-hot input
        if x.dim() == 2:
            x = x.argmax(dim=1)

        # Get embeddings
        embeddings = self.embedding(x)
        return self.norm(embeddings)

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.embedding_dim


class HierarchicalLocationEmbedder(nn.Module):
    """
    Hierarchical Location Embedder for encoding anatomical hierarchy.

    This variant encodes the hierarchical structure of digestive tract:
    - Organ level: Esophagus, Stomach, Duodenum
    - Sub-region level: Cardia, Fundus, Body, Antrum (for stomach)

    Args:
        num_organs: Number of organ categories (e.g., 3: esophagus, stomach, duodenum).
        num_subregions: Number of sub-region categories within organs.
        embedding_dim: Output embedding dimension.
        organ_dim: Dimension for organ-level embedding.
        subregion_dim: Dimension for sub-region embedding.
    """

    # Mapping from location index to (organ_idx, subregion_idx)
    # This can be customized based on the actual location categories
    DEFAULT_HIERARCHY = {
        0: (0, 0),  # Esophagus
        1: (1, 0),  # Gastric Cardia
        2: (1, 1),  # Gastric Fundus
        3: (1, 2),  # Gastric Body
        4: (1, 3),  # Gastric Antrum
        5: (2, 0),  # Duodenum
    }

    def __init__(
        self,
        num_organs: int = 3,
        num_subregions: int = 5,
        embedding_dim: int = 128,
        organ_dim: int = 64,
        subregion_dim: int = 64,
        hierarchy_map: Optional[dict] = None
    ):
        super().__init__()

        self.num_organs = num_organs
        self.num_subregions = num_subregions
        self.embedding_dim = embedding_dim
        self.hierarchy_map = hierarchy_map or self.DEFAULT_HIERARCHY

        # Separate embeddings for organ and sub-region
        self.organ_embedding = nn.Embedding(num_organs, organ_dim)
        self.subregion_embedding = nn.Embedding(num_subregions, subregion_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(organ_dim + subregion_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hierarchical embedder.

        Args:
            x: Location indices tensor of shape (B,) with integer values
               OR one-hot encoded tensor of shape (B, num_locations).

        Returns:
            Location embedding tensor of shape (B, embedding_dim).
        """
        # Handle one-hot input
        if x.dim() == 2:
            location_indices = x.argmax(dim=1)
        else:
            location_indices = x

        batch_size = location_indices.shape[0]
        device = location_indices.device

        # Get organ and sub-region indices from hierarchy
        organ_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        subregion_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i, loc_idx in enumerate(location_indices.tolist()):
            organ_idx, subregion_idx = self.hierarchy_map.get(loc_idx, (0, 0))
            organ_indices[i] = organ_idx
            subregion_indices[i] = subregion_idx

        # Get embeddings
        organ_emb = self.organ_embedding(organ_indices)
        subregion_emb = self.subregion_embedding(subregion_indices)

        # Concatenate and fuse
        combined = torch.cat([organ_emb, subregion_emb], dim=1)
        return self.fusion(combined)

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.embedding_dim


class LocationEmbedderWithPrior(nn.Module):
    """
    Location Embedder with Clinical Prior Knowledge.

    This variant incorporates clinical prior knowledge about the relationship
    between tumor types and locations (e.g., GIST is more common in stomach,
    leiomyoma is more common in esophagus).

    Args:
        num_locations: Number of organ location categories.
        num_classes: Number of tumor classes.
        embedding_dim: Output embedding dimension.
        prior_weight: Weight for prior knowledge influence.
    """

    def __init__(
        self,
        num_locations: int = 6,
        num_classes: int = 5,
        embedding_dim: int = 128,
        prior_weight: float = 0.1
    ):
        super().__init__()

        self.num_locations = num_locations
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.prior_weight = prior_weight

        # Base embedder
        self.base_embedder = nn.Sequential(
            nn.Linear(num_locations, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )

        # Prior knowledge embedding (learnable)
        # This matrix captures location-class relationships
        self.prior_matrix = nn.Parameter(
            torch.randn(num_locations, num_classes) * 0.1
        )

        # Prior fusion
        self.prior_fusion = nn.Sequential(
            nn.Linear(num_classes, embedding_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with prior knowledge.

        Args:
            x: One-hot encoded location tensor of shape (B, num_locations).

        Returns:
            Location embedding tensor of shape (B, embedding_dim).
        """
        # Base embedding
        base_emb = self.base_embedder(x)

        # Prior-based embedding
        prior_logits = torch.matmul(x, self.prior_matrix)
        prior_emb = self.prior_fusion(prior_logits)

        # Combine base and prior embeddings
        output = base_emb + self.prior_weight * prior_emb

        return output

    def get_prior_distribution(self, location_idx: int) -> torch.Tensor:
        """Get the prior class distribution for a given location."""
        return torch.softmax(self.prior_matrix[location_idx], dim=0)

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.embedding_dim


def create_location_embedder(
    num_locations: int = 6,
    embedding_dim: int = 128,
    hidden_dim: int = 64,
    variant: str = 'mlp',
    **kwargs
) -> nn.Module:
    """
    Factory function to create location embedder.

    Args:
        num_locations: Number of location categories.
        embedding_dim: Output embedding dimension.
        hidden_dim: Hidden layer dimension.
        variant: Embedder variant. Options:
            - 'linear': Simple linear layer
            - 'mlp': MLP with hidden layer
            - 'learnable': Learnable embedding table
            - 'hierarchical': Hierarchical structure encoding
            - 'prior': With clinical prior knowledge

    Returns:
        Location embedder instance.
    """
    if variant == 'linear':
        return LocationEmbedder(
            num_locations=num_locations,
            embedding_dim=embedding_dim,
            use_mlp=False
        )
    elif variant == 'mlp':
        return LocationEmbedder(
            num_locations=num_locations,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            use_mlp=True
        )
    elif variant == 'learnable':
        return LearnableLocationEmbedding(
            num_locations=num_locations,
            embedding_dim=embedding_dim
        )
    elif variant == 'hierarchical':
        return HierarchicalLocationEmbedder(
            embedding_dim=embedding_dim,
            **kwargs
        )
    elif variant == 'prior':
        return LocationEmbedderWithPrior(
            num_locations=num_locations,
            embedding_dim=embedding_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

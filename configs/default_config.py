"""
Default configuration for Trimodal Fusion Transformer Classifier (TFTC).

This configuration file contains all hyperparameters and settings for training
the TFTC model for submucosal tumor (SMT) classification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ImageConfig:
    """Configuration for image preprocessing."""
    wli_size: Tuple[int, int] = (224, 224)  # WLI image size (H, W)
    eus_size: Tuple[int, int] = (224, 224)  # EUS image size (H, W)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet mean
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet std


@dataclass
class WLIEncoderConfig:
    """Configuration for WLI (White Light Imaging) Encoder."""
    model_name: str = "swin_v2_b"  # Options: swin_v2_b, swin_v2_s, vit_l_16, vit_b_16
    pretrained: bool = True
    freeze_layers: int = 0  # Number of layers to freeze (0 = none)
    output_dim: int = 1024  # Output feature dimension


@dataclass
class EUSEncoderConfig:
    """Configuration for EUS (Endoscopic Ultrasound) Encoder."""
    model_name: str = "convnext_base"  # Options: convnext_base, convnext_small, resnet50
    pretrained: bool = True
    freeze_layers: int = 0  # Number of layers to freeze (0 = none)
    output_dim: int = 1024  # Output feature dimension


@dataclass
class LocationEmbedderConfig:
    """Configuration for Location Embedder."""
    num_locations: int = 6  # Number of organ locations
    embedding_dim: int = 128  # Location embedding dimension
    hidden_dim: int = 64  # Hidden layer dimension in MLP
    use_mlp: bool = True  # Use MLP instead of simple linear layer


@dataclass
class FusionTransformerConfig:
    """Configuration for Fusion Transformer Encoder."""
    num_layers: int = 3  # Number of transformer encoder layers
    num_heads: int = 8  # Number of attention heads
    hidden_dim: int = 2176  # Hidden dimension (2*img_dim + loc_dim = 2*1024+128)
    ff_dim: int = 4096  # Feed-forward network dimension
    dropout: float = 0.1  # Dropout rate
    attention_dropout: float = 0.1  # Attention dropout rate


@dataclass
class ClassifierConfig:
    """Configuration for Classification Head."""
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.3
    num_classes: int = 5  # Number of tumor classes


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Basic training parameters
    batch_size: int = 16
    num_epochs: int = 100
    num_workers: int = 4

    # Optimizer parameters
    optimizer: str = "adamw"  # Options: adamw, radam
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    # Learning rate scheduler
    scheduler: str = "cosine"  # Options: cosine, plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    plateau_patience: int = 10
    plateau_factor: float = 0.5

    # Loss function
    loss_fn: str = "focal"  # Options: focal, weighted_ce
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None  # Class weights for focal loss

    # Early stopping
    early_stopping_patience: int = 20

    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "checkpoints"

    # Mixed precision training
    use_amp: bool = True

    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # WLI augmentation
    wli_random_crop: bool = True
    wli_horizontal_flip: float = 0.5
    wli_vertical_flip: float = 0.3
    wli_rotation_degrees: int = 15
    wli_brightness: float = 0.2
    wli_contrast: float = 0.2
    wli_saturation: float = 0.2
    wli_hue: float = 0.1
    wli_gaussian_blur: float = 0.3
    wli_sharpness: float = 0.3

    # EUS augmentation
    eus_random_crop: bool = True
    eus_horizontal_flip: float = 0.5
    eus_vertical_flip: float = 0.3
    eus_rotation_degrees: int = 15
    eus_brightness: float = 0.15
    eus_contrast: float = 0.15
    eus_speckle_noise: float = 0.3
    eus_speckle_intensity: float = 0.1


@dataclass
class DataConfig:
    """Configuration for dataset."""
    data_root: str = "data"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    wli_dir: str = "wli_images"
    eus_dir: str = "eus_images"

    # Location categories
    location_categories: List[str] = field(default_factory=lambda: [
        "esophagus",      # 食管
        "gastric_cardia", # 贲门
        "gastric_fundus", # 胃底
        "gastric_body",   # 胃体
        "gastric_antrum", # 胃窦
        "duodenum"        # 十二指肠
    ])

    # Tumor classes
    tumor_classes: List[str] = field(default_factory=lambda: [
        "esophageal_leiomyoma",  # 食管平滑肌瘤
        "gastric_leiomyoma",     # 胃平滑肌瘤
        "gist",                   # 胃肠道间质瘤
        "lipoma",                 # 脂肪瘤
        "other"                   # 其他
    ])


@dataclass
class TFTCConfig:
    """Main configuration class combining all sub-configurations."""
    image: ImageConfig = field(default_factory=ImageConfig)
    wli_encoder: WLIEncoderConfig = field(default_factory=WLIEncoderConfig)
    eus_encoder: EUSEncoderConfig = field(default_factory=EUSEncoderConfig)
    location_embedder: LocationEmbedderConfig = field(default_factory=LocationEmbedderConfig)
    fusion_transformer: FusionTransformerConfig = field(default_factory=FusionTransformerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment settings
    experiment_name: str = "tftc_default"
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Ensure fusion transformer hidden_dim matches concatenated features
        expected_dim = (
            self.wli_encoder.output_dim +
            self.eus_encoder.output_dim +
            self.location_embedder.embedding_dim
        )
        if self.fusion_transformer.hidden_dim != expected_dim:
            self.fusion_transformer.hidden_dim = expected_dim

        # Ensure classifier num_classes matches tumor classes
        self.classifier.num_classes = len(self.data.tumor_classes)

        # Ensure location embedder num_locations matches location categories
        self.location_embedder.num_locations = len(self.data.location_categories)


def get_default_config() -> TFTCConfig:
    """Get default TFTC configuration."""
    return TFTCConfig()


def get_ablation_config(modalities: List[str]) -> TFTCConfig:
    """
    Get configuration for ablation studies.

    Args:
        modalities: List of modalities to use. Options: ["wli", "eus", "location"]

    Returns:
        TFTCConfig with adjusted settings for ablation study.
    """
    config = TFTCConfig()
    config.experiment_name = f"ablation_{'_'.join(sorted(modalities))}"

    # Adjust fusion transformer hidden_dim based on active modalities
    hidden_dim = 0
    if "wli" in modalities:
        hidden_dim += config.wli_encoder.output_dim
    if "eus" in modalities:
        hidden_dim += config.eus_encoder.output_dim
    if "location" in modalities:
        hidden_dim += config.location_embedder.embedding_dim

    config.fusion_transformer.hidden_dim = hidden_dim

    return config

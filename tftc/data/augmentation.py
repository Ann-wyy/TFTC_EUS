"""
Data Augmentation Module.

This module implements specialized data augmentation pipelines for
WLI and EUS medical images.
"""

import torch
import numpy as np
from typing import Callable, Dict, Optional, Tuple
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import random


class SpeckleNoise:
    """
    Add speckle noise to simulate ultrasound artifacts.

    Speckle noise is multiplicative noise typical in ultrasound images.
    """

    def __init__(self, intensity: float = 0.1, p: float = 0.5):
        self.intensity = intensity
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        noise = torch.randn_like(img) * self.intensity
        noisy_img = img * (1 + noise)
        return torch.clamp(noisy_img, 0, 1)


class RandomHighlight:
    """
    Add random highlight spots to simulate light reflections in endoscopy.
    """

    def __init__(
        self,
        num_spots: Tuple[int, int] = (1, 5),
        size_range: Tuple[int, int] = (5, 20),
        intensity: float = 0.3,
        p: float = 0.3
    ):
        self.num_spots = num_spots
        self.size_range = size_range
        self.intensity = intensity
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        _, h, w = img.shape
        num = random.randint(*self.num_spots)

        for _ in range(num):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            size = random.randint(*self.size_range)

            y, x = torch.meshgrid(
                torch.arange(h),
                torch.arange(w),
                indexing='ij'
            )
            dist = ((x - cx) ** 2 + (y - cy) ** 2).float().sqrt()
            mask = torch.exp(-dist / (size / 2)) * self.intensity
            mask = mask.unsqueeze(0)

            img = img + mask
            img = torch.clamp(img, 0, 1)

        return img


class ScanLineSimulation:
    """
    Simulate ultrasound scan line artifacts.
    """

    def __init__(self, intensity: float = 0.05, p: float = 0.3):
        self.intensity = intensity
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        _, h, w = img.shape

        # Create horizontal scan lines
        lines = torch.zeros((1, h, 1))
        for i in range(0, h, 2):
            lines[0, i, 0] = self.intensity * random.uniform(0.5, 1.5)

        img = img + lines.expand_as(img)
        return torch.clamp(img, 0, 1)


class ElasticDeformation:
    """
    Apply elastic deformation to simulate tissue deformation.
    """

    def __init__(
        self,
        alpha: float = 50.0,
        sigma: float = 5.0,
        p: float = 0.3
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        # This is a simplified version - full implementation would use
        # scipy.ndimage for proper elastic deformation
        return img


def get_wli_transforms(
    img_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation_config: Optional[Dict] = None
) -> Callable:
    """
    Get transform pipeline for WLI images.

    Args:
        img_size: Target image size (H, W).
        is_training: Whether this is for training (enable augmentation).
        normalize_mean: Normalization mean values.
        normalize_std: Normalization std values.
        augmentation_config: Custom augmentation parameters.

    Returns:
        Transform pipeline.
    """
    aug_cfg = augmentation_config or {}

    if is_training:
        transforms = [
            T.Resize((int(img_size[0] * 1.1), int(img_size[1] * 1.1))),
            T.RandomCrop(img_size) if aug_cfg.get('random_crop', True) else T.CenterCrop(img_size),
            T.RandomHorizontalFlip(p=aug_cfg.get('horizontal_flip', 0.5)),
            T.RandomVerticalFlip(p=aug_cfg.get('vertical_flip', 0.3)),
            T.RandomRotation(degrees=aug_cfg.get('rotation_degrees', 15)),
            T.ColorJitter(
                brightness=aug_cfg.get('brightness', 0.2),
                contrast=aug_cfg.get('contrast', 0.2),
                saturation=aug_cfg.get('saturation', 0.2),
                hue=aug_cfg.get('hue', 0.1)
            ),
            T.RandomApply([
                T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=aug_cfg.get('gaussian_blur', 0.3)),
            T.RandomAdjustSharpness(sharpness_factor=2, p=aug_cfg.get('sharpness', 0.3)),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    else:
        transforms = [
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ]

    return T.Compose(transforms)


def get_eus_transforms(
    img_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation_config: Optional[Dict] = None
) -> Callable:
    """
    Get transform pipeline for EUS images.

    Args:
        img_size: Target image size (H, W).
        is_training: Whether this is for training (enable augmentation).
        normalize_mean: Normalization mean values.
        normalize_std: Normalization std values.
        augmentation_config: Custom augmentation parameters.

    Returns:
        Transform pipeline.
    """
    aug_cfg = augmentation_config or {}

    if is_training:
        transforms = [
            T.Resize((int(img_size[0] * 1.1), int(img_size[1] * 1.1))),
            T.RandomCrop(img_size) if aug_cfg.get('random_crop', True) else T.CenterCrop(img_size),
            T.RandomHorizontalFlip(p=aug_cfg.get('horizontal_flip', 0.5)),
            T.RandomVerticalFlip(p=aug_cfg.get('vertical_flip', 0.3)),
            T.RandomRotation(degrees=aug_cfg.get('rotation_degrees', 15)),
            T.ColorJitter(
                brightness=aug_cfg.get('brightness', 0.15),
                contrast=aug_cfg.get('contrast', 0.15),
            ),
            T.ToTensor(),
        ]

        # Add EUS-specific augmentations
        tensor_transforms = []
        if aug_cfg.get('speckle_noise', 0.3) > 0:
            tensor_transforms.append(
                SpeckleNoise(
                    intensity=aug_cfg.get('speckle_intensity', 0.1),
                    p=aug_cfg.get('speckle_noise', 0.3)
                )
            )

        tensor_transforms.append(
            T.Normalize(mean=normalize_mean, std=normalize_std)
        )

        return T.Compose(transforms + tensor_transforms)
    else:
        transforms = [
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ]

    return T.Compose(transforms)


class MixUp:
    """
    MixUp augmentation for mixing two samples.

    Reference: mixup: Beyond Empirical Risk Minimization
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        wli1: torch.Tensor,
        eus1: torch.Tensor,
        loc1: torch.Tensor,
        label1: torch.Tensor,
        wli2: torch.Tensor,
        eus2: torch.Tensor,
        loc2: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mix two samples.

        Returns:
            Tuple of (mixed_wli, mixed_eus, loc1, mixed_label)
            Note: Location is not mixed as it's categorical.
        """
        lam = np.random.beta(self.alpha, self.alpha)

        mixed_wli = lam * wli1 + (1 - lam) * wli2
        mixed_eus = lam * eus1 + (1 - lam) * eus2

        # For labels (assuming soft labels or one-hot)
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_wli, mixed_eus, loc1, mixed_label


class CutMix:
    """
    CutMix augmentation for cutting and pasting image regions.

    Reference: CutMix: Regularization Strategy to Train Strong Classifiers
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _rand_bbox(
        self,
        size: Tuple[int, int, int, int],
        lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box."""
        W = size[3]
        H = size[2]

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(
        self,
        wli1: torch.Tensor,
        eus1: torch.Tensor,
        loc1: torch.Tensor,
        label1: torch.Tensor,
        wli2: torch.Tensor,
        eus2: torch.Tensor,
        loc2: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to two samples.

        Returns:
            Tuple of (mixed_wli, mixed_eus, loc1, mixed_label, lam)
        """
        lam = np.random.beta(self.alpha, self.alpha)

        # Get random bounding box
        if wli1.dim() == 3:
            wli1 = wli1.unsqueeze(0)
            wli2 = wli2.unsqueeze(0)
            eus1 = eus1.unsqueeze(0)
            eus2 = eus2.unsqueeze(0)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(wli1.size(), lam)

        # Apply cutmix to WLI
        mixed_wli = wli1.clone()
        mixed_wli[:, :, bby1:bby2, bbx1:bbx2] = wli2[:, :, bby1:bby2, bbx1:bbx2]

        # Apply cutmix to EUS
        mixed_eus = eus1.clone()
        mixed_eus[:, :, bby1:bby2, bbx1:bbx2] = eus2[:, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (wli1.size(-1) * wli1.size(-2)))

        # Mix labels
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_wli.squeeze(0), mixed_eus.squeeze(0), loc1, mixed_label, lam

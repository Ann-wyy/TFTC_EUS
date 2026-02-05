"""
数据增强模块

超声特有增强:
- 亮度抖动
- 斑点噪声
- 随机平移

白光特有增强:
- 颜色抖动
- 模拟反光 (Glare)
- 旋转/翻转
"""

import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageDraw
import random
from typing import Callable, Optional, Tuple


class SpeckleNoise:
    """
    斑点噪声 - 模拟超声图像特有的噪声

    超声图像中的斑点噪声是乘性噪声
    """

    def __init__(self, std: float = 0.05, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        noise = torch.randn_like(img) * self.std
        noisy = img * (1 + noise)
        return torch.clamp(noisy, 0, 1)


class RandomGlare:
    """
    随机反光 - 模拟内镜图像中的光源反射
    """

    def __init__(
        self,
        num_spots: Tuple[int, int] = (1, 3),
        size_range: Tuple[int, int] = (10, 30),
        intensity: float = 0.5,
        p: float = 0.3
    ):
        self.num_spots = num_spots
        self.size_range = size_range
        self.intensity = intensity
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        img = img.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size

        num = random.randint(*self.num_spots)
        for _ in range(num):
            cx = random.randint(0, width)
            cy = random.randint(0, height)
            r = random.randint(*self.size_range)

            # 创建渐变圆形
            for i in range(r, 0, -1):
                alpha = int(255 * (1 - i / r) * self.intensity)
                color = (255, 255, 255, alpha)
                draw.ellipse(
                    [cx - i, cy - i, cx + i, cy + i],
                    fill=None,
                    outline=color[:3]
                )

        return img


class RandomShift:
    """
    随机平移 - 模拟超声探头位置变化
    """

    def __init__(self, max_shift: int = 10, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        dx = random.randint(-self.max_shift, self.max_shift)
        dy = random.randint(-self.max_shift, self.max_shift)

        return TF.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0)


class GaussianNoise:
    """
    高斯噪声
    """

    def __init__(self, std: float = 0.02, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0, 1)


class RandomBrightnessJitter:
    """
    随机亮度抖动 - 用于超声图像
    """

    def __init__(self, brightness: float = 0.15, p: float = 0.5):
        self.brightness = brightness
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        factor = 1 + random.uniform(-self.brightness, self.brightness)
        return TF.adjust_brightness(img, factor)


def get_eus_transforms(
    img_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    config: Optional[dict] = None
) -> Callable:
    """
    获取超声图像变换

    Args:
        img_size: 目标图像尺寸
        is_training: 是否为训练模式
        normalize_mean: 归一化均值
        normalize_std: 归一化标准差
        config: 增强配置

    Returns:
        变换函数
    """
    cfg = config or {}

    if is_training:
        transforms = [
            T.Resize((int(img_size[0] * 1.1), int(img_size[1] * 1.1))),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=cfg.get('horizontal_flip', 0.5)),
            T.RandomVerticalFlip(p=cfg.get('vertical_flip', 0.3)),
            T.RandomRotation(degrees=cfg.get('rotation', 15)),
            RandomBrightnessJitter(
                brightness=cfg.get('brightness_jitter', 0.15),
                p=0.5
            ),
            RandomShift(
                max_shift=cfg.get('random_shift', 10),
                p=0.5
            ),
            T.ToTensor(),
        ]

        # 添加tensor级别的增强
        tensor_transforms = [
            SpeckleNoise(
                std=cfg.get('noise_std', 0.05),
                p=cfg.get('noise_prob', 0.3)
            ),
            T.Normalize(mean=normalize_mean, std=normalize_std)
        ]

        return T.Compose(transforms + tensor_transforms)

    else:
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std)
        ])


def get_wli_transforms(
    img_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    config: Optional[dict] = None
) -> Callable:
    """
    获取白光图像变换

    Args:
        img_size: 目标图像尺寸
        is_training: 是否为训练模式
        normalize_mean: 归一化均值
        normalize_std: 归一化标准差
        config: 增强配置

    Returns:
        变换函数
    """
    cfg = config or {}

    if is_training:
        transforms = [
            T.Resize((int(img_size[0] * 1.1), int(img_size[1] * 1.1))),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=cfg.get('horizontal_flip', 0.5)),
            T.RandomVerticalFlip(p=cfg.get('vertical_flip', 0.3)),
            T.RandomRotation(degrees=cfg.get('rotation', 15)),
            T.ColorJitter(
                brightness=cfg.get('brightness', 0.2),
                contrast=cfg.get('contrast', 0.2),
                saturation=cfg.get('saturation', 0.2),
                hue=cfg.get('hue', 0.1)
            ),
            RandomGlare(
                intensity=0.4,
                p=cfg.get('glare_prob', 0.3)
            ),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std)
        ]

        return T.Compose(transforms)

    else:
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std)
        ])


class MixUp:
    """
    MixUp数据增强

    用于病人级别的混合
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        eus1: torch.Tensor,
        wli1: torch.Tensor,
        label1: torch.Tensor,
        eus2: torch.Tensor,
        wli2: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        混合两个样本

        Returns:
            混合后的 (eus, wli, soft_label)
        """
        lam = np.random.beta(self.alpha, self.alpha)

        mixed_eus = lam * eus1 + (1 - lam) * eus2
        mixed_wli = lam * wli1 + (1 - lam) * wli2

        # 创建软标签
        num_classes = 6  # 固定类别数
        soft_label = torch.zeros(num_classes)
        soft_label[label1] = lam
        soft_label[label2] = 1 - lam

        return mixed_eus, mixed_wli, soft_label


class CutMix:
    """
    CutMix数据增强

    在帧级别进行裁剪和粘贴
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _rand_bbox(
        self,
        size: Tuple[int, int, int, int],
        lam: float
    ) -> Tuple[int, int, int, int]:
        """生成随机边界框"""
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
        eus1: torch.Tensor,
        wli1: torch.Tensor,
        label1: torch.Tensor,
        eus2: torch.Tensor,
        wli2: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        CutMix两个样本

        Returns:
            (mixed_eus, mixed_wli, soft_label, lam)
        """
        lam = np.random.beta(self.alpha, self.alpha)

        # 对每一帧应用CutMix
        N = eus1.shape[0]
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(eus1.shape, lam)

        mixed_eus = eus1.clone()
        mixed_wli = wli1.clone()

        mixed_eus[:, :, bby1:bby2, bbx1:bbx2] = eus2[:, :, bby1:bby2, bbx1:bbx2]
        mixed_wli[:, :, bby1:bby2, bbx1:bbx2] = wli2[:, :, bby1:bby2, bbx1:bbx2]

        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (eus1.shape[-1] * eus1.shape[-2]))

        # 软标签
        num_classes = 6
        soft_label = torch.zeros(num_classes)
        soft_label[label1] = lam
        soft_label[label2] = 1 - lam

        return mixed_eus, mixed_wli, soft_label, lam

"""
Dataset Module for TFTC.

This module implements the PyTorch Dataset class for loading trimodal
data (WLI images, EUS images, and location metadata) for SMT classification.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

from .augmentation import get_wli_transforms, get_eus_transforms


class SMTDataset(Dataset):
    """
    Dataset for Submucosal Tumor (SMT) classification.

    Loads trimodal data:
    1. WLI (White Light Imaging) images
    2. EUS (Endoscopic Ultrasound) images
    3. Location metadata (organ position)

    Expected CSV format:
        patient_id,wli_path,eus_path,location,label
        001,wli/001.jpg,eus/001.jpg,gastric_body,gist
        ...

    Args:
        data_root: Root directory containing the data.
        csv_file: Path to the CSV file with annotations.
        wli_transform: Transform pipeline for WLI images.
        eus_transform: Transform pipeline for EUS images.
        location_categories: List of location category names.
        tumor_classes: List of tumor class names.
        return_path: Whether to return image paths (for debugging).
    """

    def __init__(
        self,
        data_root: str,
        csv_file: str,
        wli_transform: Optional[Callable] = None,
        eus_transform: Optional[Callable] = None,
        location_categories: Optional[List[str]] = None,
        tumor_classes: Optional[List[str]] = None,
        return_path: bool = False
    ):
        self.data_root = Path(data_root)
        self.return_path = return_path

        # Load annotations
        csv_path = self.data_root / csv_file if not os.path.isabs(csv_file) else Path(csv_file)
        self.annotations = pd.read_csv(csv_path)

        # Set transforms
        self.wli_transform = wli_transform or get_wli_transforms(is_training=False)
        self.eus_transform = eus_transform or get_eus_transforms(is_training=False)

        # Define categories and classes
        self.location_categories = location_categories or [
            'esophagus',
            'gastric_cardia',
            'gastric_fundus',
            'gastric_body',
            'gastric_antrum',
            'duodenum'
        ]

        self.tumor_classes = tumor_classes or [
            'esophageal_leiomyoma',
            'gastric_leiomyoma',
            'gist',
            'lipoma',
            'other'
        ]

        # Create mappings
        self.location_to_idx = {loc: idx for idx, loc in enumerate(self.location_categories)}
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.tumor_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate that all required columns exist and files are accessible."""
        required_columns = ['wli_path', 'eus_path', 'location', 'label']
        missing_cols = [col for col in required_columns if col not in self.annotations.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - wli_image: Transformed WLI image tensor
                - eus_image: Transformed EUS image tensor
                - location: One-hot encoded location tensor
                - label: Class label (integer)
                - (optional) paths: Dictionary of image paths
        """
        row = self.annotations.iloc[idx]

        # Load WLI image
        wli_path = self.data_root / row['wli_path']
        wli_image = Image.open(wli_path).convert('RGB')
        wli_image = self.wli_transform(wli_image)

        # Load EUS image
        eus_path = self.data_root / row['eus_path']
        eus_image = Image.open(eus_path).convert('RGB')
        eus_image = self.eus_transform(eus_image)

        # Encode location as one-hot
        location_str = row['location'].lower().strip()
        location_idx = self.location_to_idx.get(location_str, 0)
        location_onehot = torch.zeros(len(self.location_categories))
        location_onehot[location_idx] = 1.0

        # Encode label
        label_str = row['label'].lower().strip()
        label = self.class_to_idx.get(label_str, len(self.tumor_classes) - 1)

        sample = {
            'wli_image': wli_image,
            'eus_image': eus_image,
            'location': location_onehot,
            'label': torch.tensor(label, dtype=torch.long)
        }

        if self.return_path:
            sample['paths'] = {
                'wli': str(wli_path),
                'eus': str(eus_path)
            }

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.

        Returns:
            Tensor of class weights (inverse frequency).
        """
        labels = self.annotations['label'].apply(
            lambda x: self.class_to_idx.get(x.lower().strip(), len(self.tumor_classes) - 1)
        ).values

        class_counts = np.bincount(labels, minlength=len(self.tumor_classes))
        class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
        class_weights = class_weights / class_weights.sum()  # Normalize

        return torch.tensor(class_weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights for WeightedRandomSampler.

        Returns:
            Tensor of sample weights.
        """
        class_weights = self.get_class_weights()
        labels = self.annotations['label'].apply(
            lambda x: self.class_to_idx.get(x.lower().strip(), len(self.tumor_classes) - 1)
        ).values

        sample_weights = class_weights[labels]
        return sample_weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {}
        for cls in self.tumor_classes:
            count = (self.annotations['label'].str.lower().str.strip() == cls).sum()
            distribution[cls] = count
        return distribution


class SMTDatasetWithMixup(SMTDataset):
    """
    SMT Dataset with MixUp/CutMix augmentation support.

    This variant supports mixing samples during training.
    """

    def __init__(
        self,
        data_root: str,
        csv_file: str,
        wli_transform: Optional[Callable] = None,
        eus_transform: Optional[Callable] = None,
        location_categories: Optional[List[str]] = None,
        tumor_classes: Optional[List[str]] = None,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.0
    ):
        super().__init__(
            data_root=data_root,
            csv_file=csv_file,
            wli_transform=wli_transform,
            eus_transform=eus_transform,
            location_categories=location_categories,
            tumor_classes=tumor_classes
        )

        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with optional mixing.

        Returns:
            Dictionary with potentially mixed samples and soft labels.
        """
        # Get primary sample
        sample = super().__getitem__(idx)

        # Decide whether to apply mixing
        mix_type = None
        rand = np.random.random()
        if rand < self.mixup_prob:
            mix_type = 'mixup'
        elif rand < self.mixup_prob + self.cutmix_prob:
            mix_type = 'cutmix'

        if mix_type is None:
            # Convert label to one-hot for consistency
            label_onehot = torch.zeros(len(self.tumor_classes))
            label_onehot[sample['label']] = 1.0
            sample['label'] = label_onehot
            return sample

        # Get a random second sample
        idx2 = np.random.randint(len(self))
        sample2 = super().__getitem__(idx2)

        # Create one-hot labels
        label1 = torch.zeros(len(self.tumor_classes))
        label1[sample['label']] = 1.0

        label2 = torch.zeros(len(self.tumor_classes))
        label2[sample2['label']] = 1.0

        if mix_type == 'mixup':
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            sample['wli_image'] = lam * sample['wli_image'] + (1 - lam) * sample2['wli_image']
            sample['eus_image'] = lam * sample['eus_image'] + (1 - lam) * sample2['eus_image']
            sample['label'] = lam * label1 + (1 - lam) * label2

        elif mix_type == 'cutmix':
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(sample['wli_image'].shape, lam)

            sample['wli_image'][:, bby1:bby2, bbx1:bbx2] = sample2['wli_image'][:, bby1:bby2, bbx1:bbx2]
            sample['eus_image'][:, bby1:bby2, bbx1:bbx2] = sample2['eus_image'][:, bby1:bby2, bbx1:bbx2]

            # Adjust lambda
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (sample['wli_image'].shape[-1] * sample['wli_image'].shape[-2]))
            sample['label'] = lam * label1 + (1 - lam) * label2

        return sample

    def _rand_bbox(
        self,
        shape: Tuple[int, int, int],
        lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        _, H, W = shape
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


def create_data_loaders(
    data_root: str,
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (224, 224),
    location_categories: Optional[List[str]] = None,
    tumor_classes: Optional[List[str]] = None,
    use_weighted_sampler: bool = True,
    augmentation_config: Optional[Dict] = None
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_root: Root directory containing the data.
        train_csv: Path to training CSV file.
        val_csv: Path to validation CSV file.
        test_csv: Path to test CSV file (optional).
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        img_size: Target image size.
        location_categories: List of location categories.
        tumor_classes: List of tumor classes.
        use_weighted_sampler: Use weighted random sampler for training.
        augmentation_config: Augmentation configuration dictionary.

    Returns:
        Dictionary of DataLoaders {'train': ..., 'val': ..., 'test': ...}
    """
    aug_cfg = augmentation_config or {}

    # Create transforms
    wli_train_transform = get_wli_transforms(
        img_size=img_size,
        is_training=True,
        augmentation_config=aug_cfg.get('wli')
    )
    wli_val_transform = get_wli_transforms(
        img_size=img_size,
        is_training=False
    )

    eus_train_transform = get_eus_transforms(
        img_size=img_size,
        is_training=True,
        augmentation_config=aug_cfg.get('eus')
    )
    eus_val_transform = get_eus_transforms(
        img_size=img_size,
        is_training=False
    )

    # Create datasets
    train_dataset = SMTDataset(
        data_root=data_root,
        csv_file=train_csv,
        wli_transform=wli_train_transform,
        eus_transform=eus_train_transform,
        location_categories=location_categories,
        tumor_classes=tumor_classes
    )

    val_dataset = SMTDataset(
        data_root=data_root,
        csv_file=val_csv,
        wli_transform=wli_val_transform,
        eus_transform=eus_val_transform,
        location_categories=location_categories,
        tumor_classes=tumor_classes
    )

    # Create samplers
    train_sampler = None
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    # Create data loaders
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    # Add test loader if test_csv provided
    if test_csv is not None:
        test_dataset = SMTDataset(
            data_root=data_root,
            csv_file=test_csv,
            wli_transform=wli_val_transform,
            eus_transform=eus_val_transform,
            location_categories=location_categories,
            tumor_classes=tumor_classes
        )
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return loaders


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary.
    """
    wli_images = torch.stack([s['wli_image'] for s in batch])
    eus_images = torch.stack([s['eus_image'] for s in batch])
    locations = torch.stack([s['location'] for s in batch])
    labels = torch.stack([s['label'] for s in batch])

    return {
        'wli_image': wli_images,
        'eus_image': eus_images,
        'location': locations,
        'label': labels
    }

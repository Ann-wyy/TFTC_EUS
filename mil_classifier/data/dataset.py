"""
数据集模块 - 用于MIL (Multiple Instance Learning)

每个病人是一个bag，包含多个帧
每帧包含超声图像和白光图像
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import random


class EUSPreprocessor:
    """
    超声图像预处理器

    将灰度超声图像转换为3通道:
    - 通道1: 原始灰度
    - 通道2: 边缘/梯度 (Canny或Sobel)
    - 通道3: 深度/ROI增强
    """

    def __init__(
        self,
        edge_method: str = 'canny',
        use_clahe: bool = True
    ):
        self.edge_method = edge_method
        self.use_clahe = use_clahe

        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        处理超声图像

        Args:
            image: 输入图像 (灰度或RGB)

        Returns:
            3通道图像 [H, W, 3]
        """
        # 转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 通道1: CLAHE增强的灰度图
        if self.use_clahe:
            ch1 = self.clahe.apply(gray)
        else:
            ch1 = gray

        # 通道2: 边缘检测
        if self.edge_method == 'canny':
            ch2 = cv2.Canny(gray, 50, 150)
        elif self.edge_method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            ch2 = np.sqrt(sobelx**2 + sobely**2)
            ch2 = np.clip(ch2, 0, 255).astype(np.uint8)
        else:
            ch2 = gray

        # 通道3: 深度/ROI增强 (使用形态学操作增强层次结构)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ch3 = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # 合并通道
        output = np.stack([ch1, ch2, ch3], axis=-1)

        return output


class MILDataset(Dataset):
    """
    MIL数据集

    每个病人是一个bag，包含多个帧
    每帧包含超声图像和白光图像

    CSV格式:
        patient_id,frame_id,eus_path,wli_path,label,frame_label
        001,0,eus/001_0.jpg,wli/001_0.jpg,2,1
        001,1,eus/001_1.jpg,wli/001_1.jpg,2,0
        ...

    Args:
        data_root: 数据根目录
        csv_file: CSV文件路径
        transform_eus: 超声图像变换
        transform_wli: 白光图像变换
        max_frames: 每个bag最大帧数
        class_names: 类别名称列表
    """

    def __init__(
        self,
        data_root: str,
        csv_file: str,
        transform_eus: Optional[Callable] = None,
        transform_wli: Optional[Callable] = None,
        max_frames: int = 32,
        class_names: Optional[List[str]] = None,
        eus_preprocessor: Optional[EUSPreprocessor] = None
    ):
        self.data_root = Path(data_root)
        self.max_frames = max_frames
        self.transform_eus = transform_eus
        self.transform_wli = transform_wli
        self.eus_preprocessor = eus_preprocessor or EUSPreprocessor()

        self.class_names = class_names or [
            "平滑肌瘤", "脂肪瘤", "间质瘤",
            "神经内分泌瘤", "异位胰腺", "其他"
        ]
        self.num_classes = len(self.class_names)

        # 加载并按病人分组
        csv_path = self.data_root / csv_file if not os.path.isabs(csv_file) else Path(csv_file)
        self.df = pd.read_csv(csv_path)
        self._group_by_patient()

    def _group_by_patient(self):
        """按病人ID分组"""
        self.patients = []
        self.patient_labels = []

        for patient_id, group in self.df.groupby('patient_id'):
            frames = group.to_dict('records')
            self.patients.append({
                'patient_id': patient_id,
                'frames': frames,
                'label': frames[0]['label']  # 病人级标签
            })
            self.patient_labels.append(frames[0]['label'])

        self.patient_labels = np.array(self.patient_labels)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个病人的bag

        Returns:
            字典包含:
            - eus_frames: 超声帧 [N, C, H, W]
            - wli_frames: 白光帧 [N, C, H, W]
            - label: 病人级标签
            - frame_labels: 帧级标签 [N] (如果有)
            - mask: 有效帧掩码 [N]
            - patient_id: 病人ID
        """
        patient = self.patients[idx]
        frames = patient['frames']

        # 随机采样或填充到max_frames
        if len(frames) > self.max_frames:
            # 随机采样
            indices = random.sample(range(len(frames)), self.max_frames)
            indices.sort()  # 保持顺序
            frames = [frames[i] for i in indices]
        elif len(frames) < self.max_frames:
            # 需要填充的数量
            pad_count = self.max_frames - len(frames)
            frames = frames + [frames[-1]] * pad_count  # 用最后一帧填充

        # 加载图像
        eus_list = []
        wli_list = []
        frame_labels = []
        mask = []

        for i, frame in enumerate(frames):
            # 超声图像
            eus_path = self.data_root / frame['eus_path']
            eus_img = cv2.imread(str(eus_path), cv2.IMREAD_COLOR)
            if eus_img is None:
                eus_img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                eus_img = cv2.cvtColor(eus_img, cv2.COLOR_BGR2RGB)
                # 预处理为3通道 (灰度+边缘+深度)
                eus_img = self.eus_preprocessor(eus_img)

            # 白光图像
            wli_path = self.data_root / frame['wli_path']
            wli_img = cv2.imread(str(wli_path), cv2.IMREAD_COLOR)
            if wli_img is None:
                wli_img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                wli_img = cv2.cvtColor(wli_img, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image用于transform
            eus_img = Image.fromarray(eus_img)
            wli_img = Image.fromarray(wli_img)

            # 应用变换
            if self.transform_eus:
                eus_img = self.transform_eus(eus_img)
            else:
                eus_img = torch.from_numpy(np.array(eus_img)).permute(2, 0, 1).float() / 255.0

            if self.transform_wli:
                wli_img = self.transform_wli(wli_img)
            else:
                wli_img = torch.from_numpy(np.array(wli_img)).permute(2, 0, 1).float() / 255.0

            eus_list.append(eus_img)
            wli_list.append(wli_img)

            # 帧级标签 (如果有)
            if 'frame_label' in frame:
                frame_labels.append(frame['frame_label'])
            else:
                frame_labels.append(-1)  # 无标签

            # 掩码: 原始帧为True，填充帧为False
            mask.append(i < len(patient['frames']))

        # 堆叠
        eus_frames = torch.stack(eus_list)  # [N, C, H, W]
        wli_frames = torch.stack(wli_list)  # [N, C, H, W]
        frame_labels = torch.tensor(frame_labels, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.bool)

        return {
            'eus_frames': eus_frames,
            'wli_frames': wli_frames,
            'label': torch.tensor(patient['label'], dtype=torch.long),
            'frame_labels': frame_labels,
            'mask': mask,
            'patient_id': patient['patient_id'],
            'num_frames': min(len(patient['frames']), self.max_frames)
        }

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重"""
        class_counts = np.bincount(self.patient_labels, minlength=self.num_classes)
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """计算样本权重 (用于WeightedRandomSampler)"""
        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.patient_labels]
        return sample_weights


class BalancedBatchSampler(Sampler):
    """
    平衡批次采样器

    确保每个batch中包含所有类别的样本 (如果可能)
    """

    def __init__(
        self,
        dataset: MILDataset,
        batch_size: int,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 按类别分组样本索引
        self.class_indices = {}
        for idx, label in enumerate(dataset.patient_labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        self.num_classes = len(self.class_indices)
        self.samples_per_class = max(1, batch_size // self.num_classes)

    def __iter__(self):
        # 复制索引列表以便打乱
        class_indices = {k: v.copy() for k, v in self.class_indices.items()}
        for indices in class_indices.values():
            random.shuffle(indices)

        # 生成批次
        batches = []
        while True:
            batch = []
            for class_id in class_indices:
                indices = class_indices[class_id]
                if len(indices) > 0:
                    n = min(self.samples_per_class, len(indices))
                    batch.extend(indices[:n])
                    class_indices[class_id] = indices[n:]

            if len(batch) == 0:
                break

            # 补充到batch_size
            if len(batch) < self.batch_size:
                # 从有剩余的类别中随机采样
                remaining = []
                for indices in class_indices.values():
                    remaining.extend(indices)
                if len(remaining) > 0:
                    n_extra = min(self.batch_size - len(batch), len(remaining))
                    batch.extend(random.sample(remaining, n_extra))

            random.shuffle(batch)
            batches.append(batch)

        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数

    处理不同病人帧数不同的情况
    """
    eus_frames = torch.stack([b['eus_frames'] for b in batch])
    wli_frames = torch.stack([b['wli_frames'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    frame_labels = torch.stack([b['frame_labels'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    patient_ids = [b['patient_id'] for b in batch]
    num_frames = [b['num_frames'] for b in batch]

    return {
        'eus_frames': eus_frames,
        'wli_frames': wli_frames,
        'labels': labels,
        'frame_labels': frame_labels,
        'masks': masks,
        'patient_ids': patient_ids,
        'num_frames': num_frames
    }


def create_data_loaders(
    data_root: str,
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    max_frames: int = 32,
    transform_eus_train: Optional[Callable] = None,
    transform_wli_train: Optional[Callable] = None,
    transform_eus_val: Optional[Callable] = None,
    transform_wli_val: Optional[Callable] = None,
    use_balanced_sampler: bool = True
) -> Dict[str, DataLoader]:
    """
    创建数据加载器

    Args:
        data_root: 数据根目录
        train_csv: 训练集CSV
        val_csv: 验证集CSV
        test_csv: 测试集CSV (可选)
        batch_size: 批次大小
        num_workers: 工作进程数
        max_frames: 每个bag最大帧数
        transform_*: 数据变换
        use_balanced_sampler: 是否使用平衡采样器

    Returns:
        数据加载器字典
    """
    # 创建数据集
    train_dataset = MILDataset(
        data_root=data_root,
        csv_file=train_csv,
        transform_eus=transform_eus_train,
        transform_wli=transform_wli_train,
        max_frames=max_frames
    )

    val_dataset = MILDataset(
        data_root=data_root,
        csv_file=val_csv,
        transform_eus=transform_eus_val,
        transform_wli=transform_wli_val,
        max_frames=max_frames
    )

    # 创建采样器
    if use_balanced_sampler:
        train_sampler = BalancedBatchSampler(
            train_dataset, batch_size, drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    if test_csv:
        test_dataset = MILDataset(
            data_root=data_root,
            csv_file=test_csv,
            transform_eus=transform_eus_val,
            transform_wli=transform_wli_val,
            max_frames=max_frames
        )
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    return loaders

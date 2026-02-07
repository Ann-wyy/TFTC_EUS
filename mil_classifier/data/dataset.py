"""
数据集模块 - 用于MIL (Multiple Instance Learning)

数据结构 (按类别文件夹组织):
    /rootdata/cancersort/
        ├── 平滑肌瘤/                    # 类别文件夹 (标签)
        │   ├── patient_001/
        │   │   ├── ultrasound/
        │   │   │   ├── 001.jpg
        │   │   │   └── ...
        │   │   └── white_light/
        │   │       ├── 001.jpg  (与ultrasound命名一致)
        │   │       └── ...
        │   └── patient_002/
        │       └── ...
        ├── 脂肪瘤/
        │   └── ...
        └── ...

标签自动从类别文件夹名推断，无需额外的标签文件
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import random
from sklearn.model_selection import train_test_split


class EUSPreprocessor:
    """
    超声图像预处理器

    将灰度超声图像转换为3通道:
    - 通道1: 原始灰度 (CLAHE增强)
    - 通道2: 边缘/梯度 (Canny或Sobel)
    - 通道3: 深度/ROI增强 (形态学梯度)
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


class FolderMILDataset(Dataset):
    """
    基于文件夹结构的MIL数据集 (按类别文件夹组织)

    数据结构:
        data_root/
            ├── 平滑肌瘤/              # 类别文件夹
            │   ├── patient_001/
            │   │   ├── ultrasound/
            │   │   └── white_light/
            │   └── ...
            ├── 脂肪瘤/
            │   └── ...
            └── ...

    Args:
        data_root: 数据根目录
        patient_paths: 病人完整路径列表 (相对于data_root)
        labels: 病人标签列表 (与patient_paths对应)
        ultrasound_folder: 超声图像子文件夹名称
        white_light_folder: 白光图像子文件夹名称
        transform_eus: 超声图像变换
        transform_wli: 白光图像变换
        max_frames: 每个bag最大帧数
        class_names: 类别名称列表
    """

    def __init__(
        self,
        data_root: str,
        patient_paths: List[str],
        labels: List[int],
        ultrasound_folder: str = 'ultrasound',
        white_light_folder: str = 'white_light',
        transform_eus: Optional[Callable] = None,
        transform_wli: Optional[Callable] = None,
        max_frames: int = 32,
        class_names: Optional[List[str]] = None,
        eus_preprocessor: Optional[EUSPreprocessor] = None
    ):
        self.data_root = Path(data_root)
        self.patient_paths = patient_paths  # 相对路径: "类别/病人ID"
        self.labels = labels
        self.ultrasound_folder = ultrasound_folder
        self.white_light_folder = white_light_folder
        self.max_frames = max_frames
        self.transform_eus = transform_eus
        self.transform_wli = transform_wli
        self.eus_preprocessor = eus_preprocessor or EUSPreprocessor()

        self.class_names = class_names or [
            "平滑肌瘤", "脂肪瘤", "间质瘤",
            "神经内分泌瘤", "异位胰腺", "其他"
        ]
        self.num_classes = len(self.class_names)

        # 扫描每个病人的帧
        self.patients = []
        self.patient_labels = []
        self._scan_patients()

    def _scan_patients(self):
        """扫描所有病人的帧"""
        for patient_path, label in zip(self.patient_paths, self.labels):
            patient_dir = self.data_root / patient_path
            eus_dir = patient_dir / self.ultrasound_folder
            wli_dir = patient_dir / self.white_light_folder

            if not eus_dir.exists() or not wli_dir.exists():
                print(f"警告: 病人 {patient_path} 缺少超声或白光文件夹，跳过")
                continue

            # 获取超声图像列表
            eus_files = sorted([
                f.name for f in eus_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ])

            # 获取白光图像列表
            wli_files = sorted([
                f.name for f in wli_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ])

            # 找到配对的帧 (同名文件)
            common_frames = sorted(set(eus_files) & set(wli_files))

            if len(common_frames) == 0:
                print(f"警告: 病人 {patient_path} 没有配对的帧，跳过")
                continue

            self.patients.append({
                'patient_id': patient_path,  # 保存完整路径作为ID
                'frames': common_frames,
                'label': label,
                'eus_dir': eus_dir,
                'wli_dir': wli_dir
            })
            self.patient_labels.append(label)

        self.patient_labels = np.array(self.patient_labels)
        print(f"加载了 {len(self.patients)} 个病人")

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
            - frame_labels: 帧级标签 [N] (无标签时为-1)
            - mask: 有效帧掩码 [N]
            - patient_id: 病人ID
            - num_frames: 实际帧数
        """
        patient = self.patients[idx]
        frames = patient['frames'].copy()
        original_num_frames = len(frames)

        # 随机采样或填充到max_frames
        if len(frames) > self.max_frames:
            # 随机采样
            indices = random.sample(range(len(frames)), self.max_frames)
            indices.sort()
            frames = [frames[i] for i in indices]
        elif len(frames) < self.max_frames:
            # 填充
            pad_count = self.max_frames - len(frames)
            frames = frames + [frames[-1]] * pad_count

        # 加载图像
        eus_list = []
        wli_list = []
        mask = []

        for i, frame_name in enumerate(frames):
            # 超声图像
            eus_path = patient['eus_dir'] / frame_name
            eus_img = cv2.imread(str(eus_path), cv2.IMREAD_COLOR)
            if eus_img is None:
                eus_img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                eus_img = cv2.cvtColor(eus_img, cv2.COLOR_BGR2RGB)
                # 预处理为3通道 (灰度+边缘+深度)
                eus_img = self.eus_preprocessor(eus_img)

            # 白光图像
            wli_path = patient['wli_dir'] / frame_name
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

            # 掩码: 原始帧为True，填充帧为False
            mask.append(i < original_num_frames)

        # 堆叠
        eus_frames = torch.stack(eus_list)  # [N, C, H, W]
        wli_frames = torch.stack(wli_list)  # [N, C, H, W]
        frame_labels = torch.full((len(frames),), -1, dtype=torch.long)  # 无帧级标签
        mask = torch.tensor(mask, dtype=torch.bool)

        return {
            'eus_frames': eus_frames,
            'wli_frames': wli_frames,
            'label': torch.tensor(patient['label'], dtype=torch.long),
            'frame_labels': frame_labels,
            'mask': mask,
            'patient_id': patient['patient_id'],
            'num_frames': min(original_num_frames, self.max_frames)
        }

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重 (用于处理类别不平衡)"""
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

    确保每个batch中包含各类别的样本
    """

    def __init__(
        self,
        dataset: FolderMILDataset,
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
    """自定义collate函数"""
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


def scan_data_folder(
    data_root: str,
    class_names: Optional[List[str]] = None,
    ultrasound_folder: str = 'ultrasound',
    white_light_folder: str = 'white_light'
) -> Tuple[List[str], List[int], List[str]]:
    """
    扫描数据文件夹，自动从类别文件夹结构获取病人路径和标签

    数据结构:
        data_root/
            ├── 平滑肌瘤/           # class_names[0]
            │   ├── patient_001/
            │   └── ...
            ├── 脂肪瘤/             # class_names[1]
            │   └── ...
            └── ...

    Args:
        data_root: 数据根目录
        class_names: 类别名称列表 (对应文件夹名)
            如果为None，自动从文件夹名推断
        ultrasound_folder: 超声子文件夹名
        white_light_folder: 白光子文件夹名

    Returns:
        (patient_paths, labels, detected_class_names)
        - patient_paths: 病人相对路径列表 ["类别/病人ID", ...]
        - labels: 标签列表
        - detected_class_names: 检测到的类别名称列表
    """
    data_root = Path(data_root)
    patient_paths = []
    labels = []

    # 如果没有提供class_names，自动检测
    if class_names is None:
        # 查找所有包含病人数据的类别文件夹
        detected_classes = []
        for class_dir in sorted(data_root.iterdir()):
            if not class_dir.is_dir():
                continue
            # 检查是否有子文件夹包含ultrasound/white_light
            for patient_dir in class_dir.iterdir():
                if patient_dir.is_dir():
                    eus_dir = patient_dir / ultrasound_folder
                    wli_dir = patient_dir / white_light_folder
                    if eus_dir.exists() and wli_dir.exists():
                        detected_classes.append(class_dir.name)
                        break
        class_names = sorted(detected_classes)
        print(f"自动检测到 {len(class_names)} 个类别: {class_names}")

    # 创建类别到标签的映射
    class_to_label = {name: idx for idx, name in enumerate(class_names)}

    # 遍历类别文件夹
    for class_name in class_names:
        class_dir = data_root / class_name
        if not class_dir.exists():
            print(f"警告: 类别文件夹 {class_name} 不存在，跳过")
            continue

        label = class_to_label[class_name]

        # 遍历该类别下的病人文件夹
        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            # 检查是否有超声和白光子文件夹
            eus_dir = patient_dir / ultrasound_folder
            wli_dir = patient_dir / white_light_folder

            if not eus_dir.exists() or not wli_dir.exists():
                continue

            # 相对路径: 类别/病人ID
            patient_path = f"{class_name}/{patient_dir.name}"
            patient_paths.append(patient_path)
            labels.append(label)

    print(f"共找到 {len(patient_paths)} 个病人")
    for class_name in class_names:
        count = sum(1 for p in patient_paths if p.startswith(class_name + "/"))
        print(f"  - {class_name}: {count} 个病人")

    return patient_paths, labels, class_names


def create_data_loaders_from_folder(
    data_root: str,
    ultrasound_folder: str = 'ultrasound',
    white_light_folder: str = 'white_light',
    class_names: Optional[List[str]] = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    batch_size: int = 8,
    num_workers: int = 4,
    max_frames: int = 32,
    transform_eus_train: Optional[Callable] = None,
    transform_wli_train: Optional[Callable] = None,
    transform_eus_val: Optional[Callable] = None,
    transform_wli_val: Optional[Callable] = None,
    use_balanced_sampler: bool = True,
    random_seed: int = 42
) -> Tuple[Dict[str, DataLoader], List[str]]:
    """
    从文件夹创建数据加载器 (自动从文件夹结构推断标签)

    数据结构:
        data_root/
            ├── 平滑肌瘤/           # 类别文件夹 = 标签
            │   ├── patient_001/
            │   │   ├── ultrasound/
            │   │   └── white_light/
            │   └── ...
            └── ...

    Args:
        data_root: 数据根目录
        ultrasound_folder: 超声子文件夹名
        white_light_folder: 白光子文件夹名
        class_names: 类别名称列表 (如果None则自动检测)
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        batch_size: 批次大小
        num_workers: 工作进程数
        max_frames: 每个bag最大帧数
        transform_*: 数据变换
        use_balanced_sampler: 是否使用平衡采样器
        random_seed: 随机种子

    Returns:
        (数据加载器字典, 类别名称列表)
        - {'train': ..., 'val': ..., 'test': ...}
        - class_names
    """
    # 扫描数据 (自动检测类别)
    patient_paths, labels, detected_class_names = scan_data_folder(
        data_root, class_names, ultrasound_folder, white_light_folder
    )

    # 使用检测到的类别名
    class_names = detected_class_names

    # 检查每个类别的最小样本数
    from collections import Counter
    label_counts = Counter(labels)
    min_samples_per_class = min(label_counts.values())
    num_classes = len(label_counts)

    print(f"类别分布: {dict(label_counts)}")
    print(f"每类最少样本数: {min_samples_per_class}")

    # 计算是否可以进行分层抽样
    # 分层抽样要求每个类别在每个split中至少有1个样本
    can_stratify = min_samples_per_class >= 3  # 至少3个才能分到train/val/test

    # 划分数据集
    if test_ratio > 0 and len(patient_paths) > 0:
        try:
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                patient_paths, labels,
                test_size=test_ratio,
                stratify=labels if can_stratify else None,
                random_state=random_seed
            )
        except ValueError as e:
            print(f"警告: 分层抽样失败 ({e})，使用随机抽样")
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                patient_paths, labels,
                test_size=test_ratio,
                stratify=None,
                random_state=random_seed
            )
    else:
        train_val_paths, train_val_labels = patient_paths, labels
        test_paths, test_labels = [], []

    # 再分出验证集
    if len(train_val_paths) > 0:
        actual_val_ratio = val_ratio / (1 - test_ratio) if test_ratio < 1 else val_ratio
        # 重新检查是否可以分层
        train_val_counts = Counter(train_val_labels)
        can_stratify_val = min(train_val_counts.values()) >= 2

        try:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths, train_val_labels,
                test_size=actual_val_ratio,
                stratify=train_val_labels if can_stratify_val else None,
                random_state=random_seed
            )
        except ValueError as e:
            print(f"警告: 验证集分层抽样失败 ({e})，使用随机抽样")
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths, train_val_labels,
                test_size=actual_val_ratio,
                stratify=None,
                random_state=random_seed
            )
    else:
        train_paths, train_labels = [], []
        val_paths, val_labels = [], []

    print(f"训练集: {len(train_paths)}, 验证集: {len(val_paths)}, 测试集: {len(test_paths)}")

    # 创建数据集
    train_dataset = FolderMILDataset(
        data_root=data_root,
        patient_paths=train_paths,
        labels=train_labels,
        ultrasound_folder=ultrasound_folder,
        white_light_folder=white_light_folder,
        transform_eus=transform_eus_train,
        transform_wli=transform_wli_train,
        max_frames=max_frames,
        class_names=class_names
    )

    val_dataset = FolderMILDataset(
        data_root=data_root,
        patient_paths=val_paths,
        labels=val_labels,
        ultrasound_folder=ultrasound_folder,
        white_light_folder=white_light_folder,
        transform_eus=transform_eus_val,
        transform_wli=transform_wli_val,
        max_frames=max_frames,
        class_names=class_names
    )

    # 创建采样器和加载器
    if use_balanced_sampler and len(train_dataset) > 0:
        train_sampler = BalancedBatchSampler(train_dataset, batch_size, drop_last=True)
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

    # 测试集
    if len(test_paths) > 0:
        test_dataset = FolderMILDataset(
            data_root=data_root,
            patient_paths=test_paths,
            labels=test_labels,
            ultrasound_folder=ultrasound_folder,
            white_light_folder=white_light_folder,
            transform_eus=transform_eus_val,
            transform_wli=transform_wli_val,
            max_frames=max_frames,
            class_names=class_names
        )
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    return loaders, class_names

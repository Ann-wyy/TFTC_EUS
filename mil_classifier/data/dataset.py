import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Callable, Optional, Tuple, List
import torch.nn.functional as F
import random

# --------------------------------------------
# Dataset for preprocessed EUS + WLI
# --------------------------------------------

class FolderMILDatasetPreprocessed(Dataset):
    """
    支持：
    - EUS 已经处理为 npy
    - WLI 原图 + transform (transform 在 PIL 阶段应用，包含 Normalize)
    - pad / sample frames 到 max_frames
    """

    def __init__(
        self,
        eus_root: str,
        wli_root: str,
        patient_paths: List[str],
        labels: List[int],
        max_frames: int = 15,
        img_size: Tuple[int,int] = (224,224),
        transform_wli: Optional[Callable] = None,
        wli_subfolder: str = "white_light",
        is_training: bool = False,
    ):
        self.eus_root = Path(eus_root)
        self.wli_root = Path(wli_root)
        self.patient_paths = patient_paths
        self.labels = labels
        self.max_frames = max_frames
        self.img_size = img_size
        self.transform_wli = transform_wli
        self.wli_subfolder = wli_subfolder
        self.is_training = is_training
        # ImageNet stats for fallback normalization (no transform provided)
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._norm_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.patients = self._scan_patients()

    def _scan_patients(self):
        patients = []
        for p, label in zip(self.patient_paths, self.labels):
            eus_dir = self.eus_root / p
            wli_dir = self.wli_root / p / self.wli_subfolder
            if not eus_dir.exists() or not wli_dir.exists():
                continue
            eus_files = sorted([f.name for f in eus_dir.iterdir() if f.suffix == '.npy'])
            wli_files = sorted([f.name for f in wli_dir.iterdir() if f.suffix.lower() in ['.jpg','png']])
            frames = sorted(list(set(f.replace('.npy','.jpg') for f in eus_files) & set(wli_files)))
            if not frames:
                continue
            patients.append({'id':p,'frames':frames,'label':label,'eus_dir':eus_dir,'wli_dir':wli_dir})
        return patients

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pat = self.patients[idx]
        frames = pat['frames'].copy()
        n_orig = len(frames)

        # pad or sample
        if len(frames) > self.max_frames:
            if self.is_training:
                # 训练时随机采样帧，增加多样性
                indices = sorted(random.sample(range(len(frames)), self.max_frames))
                frames = [frames[i] for i in indices]
            else:
                # 验证时取中间连续段
                start = (len(frames) - self.max_frames) // 2
                frames = frames[start:start + self.max_frames]
        elif len(frames) < self.max_frames:
            frames += [frames[-1]] * (self.max_frames - len(frames))
        mask = [i < n_orig for i in range(self.max_frames)]

        eus_list, wli_list = [], []
        for f in frames:
            # -------------------
            # 读 EUS npy
            # -------------------
            eus_path = pat['eus_dir'] / f.replace('.jpg','.npy')
            eus_tensor = torch.from_numpy(np.load(eus_path)).float()  # C x H x W
            eus_tensor = eus_tensor.unsqueeze(0)  # 1 x C x H x W
            eus_tensor = F.interpolate(eus_tensor, size=self.img_size, mode='bilinear', align_corners=False)
            eus_tensor = eus_tensor.squeeze(0)    # C x H x W
            eus_list.append(eus_tensor)

            # -------------------
            # 读 WLI
            # -------------------
            wli_path = pat['wli_dir'] / f
            wli_img = Image.open(wli_path).convert('RGB')

            if self.transform_wli:
                # transform 在 PIL 阶段完成 resize / augment / ToTensor / Normalize
                wli_tensor = self.transform_wli(wli_img)
            else:
                # 无 transform 时手动做 resize + /255 + ImageNet normalize
                wli_img = wli_img.resize(self.img_size)
                wli_tensor = torch.from_numpy(np.array(wli_img)).permute(2, 0, 1).float() / 255.0
                wli_tensor = (wli_tensor - self._norm_mean) / self._norm_std

            wli_list.append(wli_tensor)

        return {
            'eus_frames': torch.stack(eus_list),
            'wli_frames': torch.stack(wli_list),
            'labels': torch.tensor(pat['label'], dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'patient_id': pat['id'],
            'num_frames': min(n_orig, self.max_frames)
        }

# --------------------------------------------
# collate_fn
# --------------------------------------------
def collate_fn(batch):
    return {
        'eus_frames': torch.stack([b['eus_frames'] for b in batch]),
        'wli_frames': torch.stack([b['wli_frames'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'patient_ids': [b['patient_id'] for b in batch],
        'num_frames': [b['num_frames'] for b in batch]
    }



def scan_data_folder(data_root: str) -> Tuple[List[str], List[int], List[str]]:
    """
    扫描数据目录，假设目录结构为:
    data_root/
        class_A/
            patient_001/
            patient_002/
        class_B/
            patient_003/
            ...

    Returns:
        paths: 所有病例文件夹相对于类别文件夹的路径列表 (例如: 'class_A/patient_001')
        labels: 对应的类别索引列表 [0, 0, 1, ...]
        class_names: 类别名称列表 ['class_A', 'class_B']
    """
    root = Path(data_root)

    # 获取所有子目录作为类别名称，并排序以保证 label 映射稳定
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])

    paths = []
    labels = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = root / class_name

        for patient_dir in class_dir.iterdir():
            if patient_dir.is_dir():
                relative_path = patient_dir.relative_to(root).as_posix()
                paths.append(relative_path)
                labels.append(label_idx)

    print(f"Successfully scanned {len(class_names)} classes: {class_names}")
    print(f"Total patients found: {len(paths)}")

    return paths, labels, class_names

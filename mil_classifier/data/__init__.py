"""
数据模块

导出数据集和增强组件

支持文件夹结构数据:
    /rootdata/cancersort/
        ├── patient_name/
        │   ├── ultrasound/
        │   │   └── *.jpg
        │   └── white_light/
        │       └── *.jpg (与ultrasound命名一致)
"""

from .dataset import (
    FolderMILDatasetPreprocessed,
    collate_fn,
    scan_data_folder,
)

from .augmentation import (
    SpeckleNoise,
    RandomGlare,
    RandomShift,
    GaussianNoise,
    RandomBrightnessJitter,
    get_eus_transforms,
    get_wli_transforms,
    MixUp,
    CutMix
)

__all__ = [
    # Dataset
    'FolderMILDatasetPreprocessed',
    'collate_fn',
    'scan_data_folder',

    # Augmentation
    'SpeckleNoise',
    'RandomGlare',
    'RandomShift',
    'GaussianNoise',
    'RandomBrightnessJitter',
    'get_eus_transforms',
    'get_wli_transforms',
    'MixUp',
    'CutMix',
]

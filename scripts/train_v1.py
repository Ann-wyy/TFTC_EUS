#!/usr/bin/env python3
"""
Multi-Modal MIL Tumor Classifier Training - 简单 8:2 划分版
(数据量小时比 K-Fold 更快验证，避免折间方差干扰)

与 train.py 的区别:
- 不做 K-Fold，直接 StratifiedShuffleSplit 8:2
- set_seed() 保证完全可复现
- 其余流程（模型/损失/优化器/early stopping/best-model 重载）完全一致
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, \
    classification_report, confusion_matrix

from mil_classifier.models import MultiModalMILClassifier
from mil_classifier.data.dataset import FolderMILDatasetPreprocessed, \
    collate_fn, scan_data_folder
from mil_classifier.data.augmentation import get_wli_transforms
from mil_classifier.losses.losses import create_loss_function
from mil_classifier.utils import EarlyStopping
from configs.config import Config, get_config


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

def set_seed(seed: int = 42):
    """全局固定随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------

def setup_logging(log_filename: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )
    return logger


# -----------------------------------------------------------------------
# Metric Logger
# -----------------------------------------------------------------------

class MetricLogger:
    def __init__(self, class_names=None, average='macro'):
        self.patient_logits_list = []
        self.patient_labels_list = []
        self.loss_total_list = []
        self.loss_patient_list = []
        self.loss_instance_list = []
        self.class_names = class_names
        self.average = average

    def update(self, patient_logits, patient_labels, loss_dict=None):
        self.patient_logits_list.append(patient_logits.detach().cpu())
        self.patient_labels_list.append(patient_labels.detach().cpu())
        if loss_dict:
            self.loss_total_list.append(loss_dict.get('total', 0))
            self.loss_patient_list.append(loss_dict.get('patient', 0))
            self.loss_instance_list.append(loss_dict.get('instance', 0))

    def compute(self):
        patient_logits = torch.cat(self.patient_logits_list, dim=0)
        patient_labels = torch.cat(self.patient_labels_list, dim=0)
        preds   = torch.argmax(patient_logits, dim=-1).numpy()
        targets = patient_labels.numpy()

        acc       = (preds == targets).mean()
        precision = precision_score(targets, preds, average=self.average, zero_division=0)
        recall    = recall_score(targets, preds, average=self.average, zero_division=0)
        f1        = f1_score(targets, preds, average=self.average, zero_division=0)

        per_class_report = None
        if self.class_names:
            report = classification_report(
                targets, preds, target_names=self.class_names,
                zero_division=0, output_dict=True,
            )
            cm = confusion_matrix(targets, preds, labels=list(range(len(self.class_names))))
            per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
            for i, name in enumerate(self.class_names):
                if name in report:
                    report[name]['accuracy'] = float(per_class_acc[i])
            per_class_report = report

        return {
            'accuracy':     acc,
            'precision':    precision,
            'recall':       recall,
            'f1':           f1,
            'per_class':    per_class_report,
            'loss_total':   np.mean(self.loss_total_list)   if self.loss_total_list   else 0,
            'loss_patient': np.mean(self.loss_patient_list) if self.loss_patient_list else 0,
            'loss_instance':np.mean(self.loss_instance_list)if self.loss_instance_list else 0,
        }

    def reset(self):
        self.patient_logits_list.clear()
        self.patient_labels_list.clear()
        self.loss_total_list.clear()
        self.loss_patient_list.clear()
        self.loss_instance_list.clear()


def log_per_class_metrics(metrics: dict, phase: str, logger: logging.Logger):
    per_class = metrics.get('per_class')
    if not per_class:
        return
    skip  = {'accuracy', 'macro avg', 'weighted avg'}
    header = f"{'Class':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>8}"
    rows  = [header, '-' * len(header)]
    for cls, vals in per_class.items():
        if cls in skip or not isinstance(vals, dict):
            continue
        rows.append(
            f"{cls:<25} "
            f"{vals.get('accuracy', 0.0):>7.3f} "
            f"{vals['precision']:>7.3f} "
            f"{vals['recall']:>7.3f} "
            f"{vals['f1-score']:>7.3f} "
            f"{int(vals['support']):>8}"
        )
    macro = per_class.get('macro avg', {})
    rows.append(
        f"{'[macro avg]':<25} "
        f"{'':>7} "
        f"{macro.get('precision', 0.0):>7.3f} "
        f"{macro.get('recall', 0.0):>7.3f} "
        f"{macro.get('f1-score', 0.0):>7.3f} "
        f"{int(macro.get('support', 0)):>8}"
    )
    logger.info(f"\n[{phase}] Per-class metrics:\n" + "\n".join(rows))


# -----------------------------------------------------------------------
# 8:2 DataLoader 构建
# -----------------------------------------------------------------------

def create_train_val_loaders(config: Config, val_ratio: float = 0.2, seed: int = 42):
    """StratifiedShuffleSplit 8:2，保持类别比例"""
    paths, labels, class_names = scan_data_folder(config.data.wli_root)
    labels_np = np.array(labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(paths, labels_np))

    aug_cfg = {
        'horizontal_flip': config.augmentation.random_horizontal_flip,
        'vertical_flip':   config.augmentation.random_vertical_flip,
        'rotation':        config.augmentation.random_rotation,
        'brightness':      config.augmentation.wli_brightness,
        'contrast':        config.augmentation.wli_contrast,
        'saturation':      config.augmentation.wli_saturation,
        'hue':             config.augmentation.wli_hue,
    }
    train_transform = get_wli_transforms(
        img_size=config.data.img_size, is_training=True,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
        config=aug_cfg,
    )
    val_transform = get_wli_transforms(
        img_size=config.data.img_size, is_training=False,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )

    train_ds = FolderMILDatasetPreprocessed(
        eus_root=config.data.eus_root,
        wli_root=config.data.wli_root,
        patient_paths=[paths[i] for i in train_idx],
        labels=[labels[i] for i in train_idx],
        max_frames=config.data.max_frames,
        img_size=config.data.img_size,
        transform_wli=train_transform,
        wli_subfolder=config.data.white_light_folder,
        is_training=True,
    )
    val_ds = FolderMILDatasetPreprocessed(
        eus_root=config.data.eus_root,
        wli_root=config.data.wli_root,
        patient_paths=[paths[i] for i in val_idx],
        labels=[labels[i] for i in val_idx],
        max_frames=config.data.max_frames,
        img_size=config.data.img_size,
        transform_wli=val_transform,
        wli_subfolder=config.data.white_light_folder,
        is_training=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True,
        num_workers=config.training.num_workers, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, collate_fn=collate_fn,
    )

    train_labels = [labels[i] for i in train_idx]
    return train_loader, val_loader, class_names, train_labels


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(config: Config, train_loader, val_loader, class_names,
          train_labels, logger, log_dir: Path):

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # 模型
    model = MultiModalMILClassifier(
        backbone=config.encoder.backbone,
        feature_dim=config.encoder.feature_dim,
        num_classes=len(class_names),
        pretrained=True,
        eus_channels=config.data.eus_channels,
        use_relational_attention=config.encoder.use_relational_attention,
    ).to(device)

    # Class weights（基于训练集分布）
    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(class_names)),
        y=np.array(train_labels),
    )
    logger.info("Class weights: " +
                ", ".join(f"{class_names[i]}={w:.3f}" for i, w in enumerate(cw)))

    criterion = create_loss_function(
        loss_type=config.training.loss_type,
        num_classes=len(class_names),
        class_weights=cw.tolist(),
        gamma=config.training.focal_gamma,
        auxiliary_weight=config.classifier.auxiliary_weight,
        use_auxiliary=False,
        label_smoothing=config.training.label_smoothing,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler     = CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    scaler        = GradScaler() if config.training.use_amp else None
    early_stopping = EarlyStopping(patience=config.training.early_stopping_patience, mode='max')

    best_acc = 0.0
    log_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = log_dir / 'best_model.pth'

    for epoch in range(config.training.num_epochs):

        # ---- Train ----
        model.train()
        train_logger = MetricLogger(class_names)
        for batch in train_loader:
            eus_frames = batch['eus_frames'].to(device)
            wli_frames = batch['wli_frames'].to(device)
            labels     = batch['labels'].to(device)
            masks      = batch['mask'].to(device)

            optimizer.zero_grad()
            if scaler:
                with autocast(device_type='cuda'):
                    outputs        = model(eus_frames, wli_frames, masks)
                    patient_logits = outputs['patient_logits']
                    loss_dict      = criterion(patient_logits, labels)
                    loss_total     = loss_dict['total']
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs        = model(eus_frames, wli_frames, masks)
                patient_logits = outputs['patient_logits']
                loss_dict      = criterion(patient_logits, labels)
                loss_total     = loss_dict['total']
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()

            train_logger.update(patient_logits, labels, {
                'total':    loss_dict['total'].item(),
                'patient':  loss_dict['patient'].item(),
                'instance': loss_dict['frame'].item(),
            })

        train_metrics = train_logger.compute()

        # ---- Val ----
        model.eval()
        val_logger = MetricLogger(class_names)
        with torch.no_grad():
            for batch in val_loader:
                eus_frames = batch['eus_frames'].to(device)
                wli_frames = batch['wli_frames'].to(device)
                labels     = batch['labels'].to(device)
                masks      = batch['mask'].to(device)
                outputs        = model(eus_frames, wli_frames, masks)
                patient_logits = outputs['patient_logits']
                loss_dict      = criterion(patient_logits, labels)
                val_logger.update(patient_logits, labels, {
                    'total':    loss_dict['total'].item(),
                    'patient':  loss_dict['patient'].item(),
                    'instance': loss_dict['frame'].item(),
                })

        val_metrics = val_logger.compute()
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1:03d}/{config.training.num_epochs} | "
            f"Train Loss {train_metrics['loss_total']:.4f} "
            f"Acc {train_metrics['accuracy']*100:.1f}% "
            f"F1 {train_metrics['f1']:.4f} | "
            f"Val Loss {val_metrics['loss_total']:.4f} "
            f"Acc {val_metrics['accuracy']*100:.1f}% "
            f"P {val_metrics['precision']:.4f} "
            f"R {val_metrics['recall']:.4f} "
            f"F1 {val_metrics['f1']:.4f}"
        )
        log_per_class_metrics(train_metrics, f"Train Epoch {epoch+1}", logger)
        log_per_class_metrics(val_metrics,   f"Val   Epoch {epoch+1}", logger)

        # 保存最佳模型
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"  -> New best val acc: {best_acc*100:.2f}%  (saved)")

        if early_stopping(epoch, val_metrics['accuracy']):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # ---- 加载最佳模型做最终评估 ----
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        logger.info(f"Loaded best checkpoint: {best_ckpt}")

    model.eval()
    final_logger = MetricLogger(class_names)
    with torch.no_grad():
        for batch in val_loader:
            eus_frames = batch['eus_frames'].to(device)
            wli_frames = batch['wli_frames'].to(device)
            labels     = batch['labels'].to(device)
            masks      = batch['mask'].to(device)
            outputs        = model(eus_frames, wli_frames, masks)
            patient_logits = outputs['patient_logits']
            loss_dict      = criterion(patient_logits, labels)
            final_logger.update(patient_logits, labels, {
                'total':    loss_dict['total'].item(),
                'patient':  loss_dict['patient'].item(),
                'instance': loss_dict['frame'].item(),
            })

    final_metrics = final_logger.compute()
    logger.info(
        f"\n[FINAL - BEST MODEL] "
        f"Val Loss {final_metrics['loss_total']:.4f} "
        f"Acc {final_metrics['accuracy']*100:.2f}% "
        f"P {final_metrics['precision']:.4f} "
        f"R {final_metrics['recall']:.4f} "
        f"F1 {final_metrics['f1']:.4f}"
    )
    log_per_class_metrics(final_metrics, "FINAL BEST", logger)
    return final_metrics


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    config = get_config()

    parser = argparse.ArgumentParser(description="Train Multi-Modal MIL (8:2 split)")
    parser.add_argument('--wli_root',   type=str,   default=config.data.wli_root)
    parser.add_argument('--eus_root',   type=str,   default=config.data.eus_root)
    parser.add_argument('--batch_size', type=int,   default=config.training.batch_size)
    parser.add_argument('--epochs',     type=int,   default=config.training.num_epochs)
    parser.add_argument('--log_dir',    type=str,   default=config.training.log_dir)
    parser.add_argument('--device',     type=str,   default=config.device)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--val_ratio',  type=float, default=0.2,
                        help='验证集比例，默认 0.2（即 8:2）')
    args = parser.parse_args()

    # 更新配置
    config.data.wli_root          = args.wli_root
    config.data.eus_root          = args.eus_root
    config.training.batch_size    = args.batch_size
    config.training.num_epochs    = args.epochs
    config.training.log_dir       = args.log_dir
    config.device                 = args.device

    # 固定随机种子
    set_seed(args.seed)

    log_dir = Path(config.training.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        str(log_dir / f"train_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    )
    logger.info(f"Seed: {args.seed} | Val ratio: {args.val_ratio}")

    train_loader, val_loader, class_names, train_labels = create_train_val_loaders(
        config, val_ratio=args.val_ratio, seed=args.seed,
    )
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(
        f"Train: {len(train_loader.dataset)} patients | "
        f"Val: {len(val_loader.dataset)} patients"
    )

    metrics = train(config, train_loader, val_loader, class_names,
                    train_labels, logger, log_dir)

    logger.info(
        f"\n{'='*60}\n"
        f"Final Val Acc  : {metrics['accuracy']*100:.2f}%\n"
        f"Final Val F1   : {metrics['f1']:.4f}\n"
        f"{'='*60}"
    )


if __name__ == '__main__':
    main()

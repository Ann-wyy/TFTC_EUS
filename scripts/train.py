#!/usr/bin/env python3
"""
SOTA Multi-Modal MIL Tumor Classifier Training (with preprocessed EUS .npy)
- Uses FolderMILDatasetPreprocessed
- Supports 5-Fold Cross Validation
- Top-K CLAM MIL pooling
- Frame Transformer
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from datetime import datetime
import logging
import random
import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from mil_classifier.models import MultiModalMILClassifier
from mil_classifier.data.dataset import FolderMILDatasetPreprocessed, collate_fn
from mil_classifier.data.augmentation import get_wli_transforms
from mil_classifier.losses.losses import create_loss_function
from mil_classifier.utils import EarlyStopping
from configs.config import Config, get_config

# ----------------------------
# Logging setup
# ----------------------------
def setup_logging(log_filename: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logger

# ----------------------------
# Metric Logger
# ----------------------------
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
        from sklearn.metrics import confusion_matrix
        patient_logits = torch.cat(self.patient_logits_list, dim=0)
        patient_labels = torch.cat(self.patient_labels_list, dim=0)
        preds = torch.argmax(patient_logits, dim=-1).numpy()
        targets = patient_labels.numpy()

        acc = (preds == targets).mean()
        precision = precision_score(targets, preds, average=self.average, zero_division=0)
        recall = recall_score(targets, preds, average=self.average, zero_division=0)
        f1 = f1_score(targets, preds, average=self.average, zero_division=0)

        per_class_report = None
        if self.class_names:
            report = classification_report(
                targets, preds, target_names=self.class_names,
                zero_division=0, output_dict=True
            )
            # 为每个类别补充每类 Acc = TP / (该类总样本数)
            cm = confusion_matrix(targets, preds, labels=list(range(len(self.class_names))))
            per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
            for i, name in enumerate(self.class_names):
                if name in report:
                    report[name]['accuracy'] = float(per_class_acc[i])
            per_class_report = report

        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class': per_class_report,
            'loss_total': np.mean(self.loss_total_list) if self.loss_total_list else 0,
            'loss_patient': np.mean(self.loss_patient_list) if self.loss_patient_list else 0,
            'loss_instance': np.mean(self.loss_instance_list) if self.loss_instance_list else 0,
        }
        return metrics

    def reset(self):
        self.patient_logits_list.clear()
        self.patient_labels_list.clear()
        self.loss_total_list.clear()
        self.loss_patient_list.clear()
        self.loss_instance_list.clear()

# ----------------------------
# Per-class metric logging
# ----------------------------
def log_per_class_metrics(metrics: dict, phase: str, logger: logging.Logger):
    """将每类的 Acc / Precision / Recall / F1 以对齐表格形式写入 logger。"""
    per_class = metrics.get('per_class')
    if not per_class:
        return
    # 跳过 sklearn 自动附加的 avg 行和 accuracy 标量
    skip = {'accuracy', 'macro avg', 'weighted avg'}
    header = f"{'Class':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>8}"
    rows = [header, '-' * len(header)]
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
    # macro 平均行
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


# ----------------------------
# K-Fold Loader
# ----------------------------
def create_kfold_loaders(config, k=5, random_seed=42):
    from mil_classifier.data.dataset import scan_data_folder
    wli_root = config.data.wli_root
    paths, labels, class_names = scan_data_folder(wli_root)
    labels_np = np.array(labels)

    aug_cfg = {
        'horizontal_flip': config.augmentation.random_horizontal_flip,
        'vertical_flip': config.augmentation.random_vertical_flip,
        'rotation': config.augmentation.random_rotation,
        'brightness': config.augmentation.wli_brightness,
        'contrast': config.augmentation.wli_contrast,
        'saturation': config.augmentation.wli_saturation,
        'hue': config.augmentation.wli_hue,
    }
    train_transform = get_wli_transforms(
        img_size=config.data.img_size,
        is_training=True,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
        config=aug_cfg,
    )
    val_transform = get_wli_transforms(
        img_size=config.data.img_size,
        is_training=False,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    loaders_per_fold = []

    for train_idx, val_idx in skf.split(paths, labels_np):
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
            num_workers=config.training.num_workers, collate_fn=collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.training.batch_size, shuffle=False,
            num_workers=config.training.num_workers, collate_fn=collate_fn
        )
        loaders_per_fold.append({
            'train': train_loader,
            'val': val_loader,
            'train_labels': [labels[i] for i in train_idx],
        })

    return loaders_per_fold, class_names

# ----------------------------
# Fold Training
# ----------------------------
def train_fold(config: Config, fold_idx, train_loader, val_loader, class_names, logger, train_labels=None):
    from sklearn.utils.class_weight import compute_class_weight
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n===== Fold {fold_idx+1} / 5 ===== | Device: {device}")

    model = MultiModalMILClassifier(
        backbone=config.encoder.backbone,
        feature_dim=config.encoder.feature_dim,
        num_classes=len(class_names),
        pretrained=True,
        eus_channels=config.data.eus_channels,
    ).to(device)

    # 计算 class weights（基于当前折训练集样本分布）
    if train_labels is not None:
        train_labels_np = np.array(train_labels)
        cw = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(len(class_names)),
            y=train_labels_np
        )
        class_weights = cw.tolist()
        logger.info(f"Fold {fold_idx+1} class_weights: " +
                    ", ".join(f"{class_names[i]}={w:.3f}" for i, w in enumerate(class_weights)))
    else:
        class_weights = config.training.class_weights

    # 使用配置的 loss_type（focal / class_balanced / ce），关闭无帧级标签的辅助任务
    criterion = create_loss_function(
        loss_type=config.training.loss_type,
        num_classes=len(class_names),
        class_weights=class_weights,
        gamma=config.training.focal_gamma,
        auxiliary_weight=config.classifier.auxiliary_weight,
        use_auxiliary=False,  # 无帧级标注，关闭辅助任务避免错误监督
    )

    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    scaler = GradScaler() if config.training.use_amp else None
    early_stopping = EarlyStopping(patience=config.training.early_stopping_patience, mode='max')

    best_acc = 0.0
    log_dir = Path(config.training.log_dir) / f"fold_{fold_idx+1}"
    log_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.training.num_epochs):
        model.train()
        metric_logger = MetricLogger(class_names)
        for batch in train_loader:
            eus_frames = batch['eus_frames'].to(device)
            wli_frames = batch['wli_frames'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            if scaler:
                with autocast(device_type='cuda'):
                    outputs = model(eus_frames, wli_frames, masks)
                    patient_logits = outputs['patient_logits']
                    loss_dict = criterion(patient_logits, labels)
                    loss_total = loss_dict['total']
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(eus_frames, wli_frames, masks)
                patient_logits = outputs['patient_logits']
                loss_dict = criterion(patient_logits, labels)
                loss_total = loss_dict['total']
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()

            metric_logger.update(patient_logits, labels, {
                'total': loss_dict['total'].item(),
                'patient': loss_dict['patient'].item(),
                'instance': loss_dict['frame'].item(),
            })

        train_metrics = metric_logger.compute()
        metric_logger.reset()

        # Validation
        model.eval()
        metric_logger = MetricLogger(class_names)
        with torch.no_grad():
            for batch in val_loader:
                eus_frames = batch['eus_frames'].to(device)
                wli_frames = batch['wli_frames'].to(device)
                labels = batch['labels'].to(device)
                masks = batch['mask'].to(device)
                outputs = model(eus_frames, wli_frames, masks)
                patient_logits = outputs['patient_logits']
                loss_dict = criterion(patient_logits, labels)
                metric_logger.update(patient_logits, labels, {
                    'total': loss_dict['total'].item(),
                    'patient': loss_dict['patient'].item(),
                    'instance': loss_dict['frame'].item(),
                })

        val_metrics = metric_logger.compute()
        metric_logger.reset()
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{config.training.num_epochs} | "
            f"Train Loss {train_metrics['loss_total']:.4f} "
            f"Acc {train_metrics['accuracy']*100:.2f}% "
            f"F1 {train_metrics['f1']:.4f} | "
            f"Val Loss {val_metrics['loss_total']:.4f} "
            f"Acc {val_metrics['accuracy']*100:.2f}% "
            f"P {val_metrics['precision']:.4f} "
            f"R {val_metrics['recall']:.4f} "
            f"F1 {val_metrics['f1']:.4f}"
        )
        log_per_class_metrics(train_metrics, f"Train Epoch {epoch+1}", logger)
        log_per_class_metrics(val_metrics, f"Val   Epoch {epoch+1}", logger)

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), log_dir / 'best_model.pth')

        if early_stopping(epoch, val_metrics['accuracy']):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    return best_acc

# ----------------------------
# Main
# ----------------------------
def main():
    config = get_config()
    parser = argparse.ArgumentParser(description="Train Multi-Modal MIL classifier")
    parser.add_argument('--wli_root', type=str, default=config.data.wli_root)
    parser.add_argument('--eus_root', type=str, default=config.data.eus_root)
    parser.add_argument('--batch_size', type=int, default=config.training.batch_size)
    parser.add_argument('--epochs', type=int, default=config.training.num_epochs)
    parser.add_argument('--log_dir', type=str, default=config.training.log_dir)
    parser.add_argument('--device', type=str, default=config.device)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config.data.wli_root = args.wli_root
    config.data.eus_root = args.eus_root
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.log_dir = args.log_dir
    config.device = args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = Path(config.training.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"))

    loaders_per_fold, class_names = create_kfold_loaders(config=config, k=4, random_seed=args.seed)

    fold_accuracies = []
    for fold_idx, loaders in enumerate(loaders_per_fold):
        logger.info(f"\n===== Training Fold {fold_idx+1} =====")
        train_loader = loaders['train']
        val_loader = loaders['val']
        acc = train_fold(config, fold_idx, train_loader, val_loader, class_names, logger,
                         train_labels=loaders.get('train_labels'))
        fold_accuracies.append(acc)
        logger.info(f"Fold {fold_idx+1} Accuracy: {acc*100:.2f}%")

    logger.info(f"\n5-Fold Average Accuracy: {np.mean(fold_accuracies)*100:.2f}%")

if __name__ == '__main__':
    main()

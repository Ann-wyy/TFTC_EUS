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
            per_class_report = classification_report(targets, preds, target_names=self.class_names, zero_division=0, output_dict=True)

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
# K-Fold Loader
# ----------------------------
def create_kfold_loaders(config, k=5, random_seed=42):
    from mil_classifier.data.dataset import scan_data_folder
    wli_root = config.data.wli_root
    paths, labels, class_names = scan_data_folder(wli_root)
    labels_np = np.array(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    loaders_per_fold = []

    for train_idx, val_idx in skf.split(paths, labels_np):
        train_ds = FolderMILDatasetPreprocessed(
            eus_root=config.data.eus_root,
            wli_root=config.data.wli_root,
            patient_paths=[paths[i] for i in train_idx],
            labels=[labels[i] for i in train_idx],
            max_frames=config.data.max_frames,
            img_size=config.data.img_size
        )
        val_ds = FolderMILDatasetPreprocessed(
            eus_root=config.data.eus_root,
            wli_root=config.data.wli_root,
            patient_paths=[paths[i] for i in val_idx],
            labels=[labels[i] for i in val_idx],
            max_frames=config.data.max_frames,
            img_size=config.data.img_size
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.training.batch_size, shuffle=True,
            num_workers=config.training.num_workers, collate_fn=collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.training.batch_size, shuffle=False,
            num_workers=config.training.num_workers, collate_fn=collate_fn
        )
        loaders_per_fold.append({'train': train_loader, 'val': val_loader})

    return loaders_per_fold, class_names

# ----------------------------
# Fold Training
# ----------------------------
def train_fold(config: Config, fold_idx, train_loader, val_loader, class_names, logger):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n===== Fold {fold_idx+1} / 5 ===== | Device: {device}")

    model = MultiModalMILClassifier(
        backbone=config.encoder.backbone,
        feature_dim=config.encoder.feature_dim,
        num_classes=len(class_names),
        pretrained=True,
        eus_channels=config.data.eus_channels,
    ).to(device)

    criterion_bag = nn.CrossEntropyLoss()
    criterion_instance = nn.CrossEntropyLoss()
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
            print("-" * 30)
            print(f"EUS Tensor 原始形状: {eus_frames.shape}")
            B, N, Ce, H, W = eus_frames.shape
            print(f"传入的参数 - B: {B}, N: {N}, Ce: {Ce}, H: {H}, W: {W}")
            print(f"预期总量: {B * N * Ce * H * W}")
            print(f"实际总量: {eus_frames.numel()}")
            print("-" * 30)

            optimizer.zero_grad()
            if scaler:
                with autocast(device_type='cuda'):
                    outputs = model(eus_frames, wli_frames, masks)
                    patient_logits = outputs['patient_logits']
                    instance_logits = outputs.get('instance_logits')
                    loss_bag = criterion_bag(patient_logits, labels)
                    loss_instance = torch.tensor(0.0, device=device)
                    if instance_logits is not None:
                        B, N, C = instance_logits.shape
                        instance_labels = labels.unsqueeze(1).repeat(1, N).view(-1)
                        mask_flat = masks.view(-1)
                        loss_instance = criterion_instance(instance_logits.view(B*N, C)[mask_flat], instance_labels[mask_flat])
                    loss_total = loss_bag + config.classifier.auxiliary_weight * loss_instance
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(eus_frames, wli_frames, masks)
                patient_logits = outputs['patient_logits']
                instance_logits = outputs.get('instance_logits')
                loss_bag = criterion_bag(patient_logits, labels)
                loss_instance = torch.tensor(0.0, device=device)
                if instance_logits is not None:
                    B, N, C = instance_logits.shape
                    instance_labels = labels.unsqueeze(1).repeat(1, N).view(-1)
                    mask_flat = masks.view(-1)
                    loss_instance = criterion_instance(instance_logits.view(B*N, C)[mask_flat], instance_labels[mask_flat])
                loss_total = loss_bag + config.classifier.auxiliary_weight * loss_instance
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()

            metric_logger.update(patient_logits, labels,
                                 {'total': loss_total.item(), 'patient': loss_bag.item(), 'instance': loss_instance.item()})

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
                instance_logits = outputs.get('instance_logits')
                loss_bag = criterion_bag(patient_logits, labels)
                loss_instance = torch.tensor(0.0, device=device)
                if instance_logits is not None:
                    B, N, C = instance_logits.shape
                    instance_labels = labels.unsqueeze(1).repeat(1, N).view(-1)
                    mask_flat = masks.view(-1)
                    loss_instance = criterion_instance(instance_logits.view(B*N, C)[mask_flat], instance_labels[mask_flat])
                loss_total = loss_bag + config.classifier.auxiliary_weight * loss_instance
                metric_logger.update(patient_logits, labels,
                                     {'total': loss_total.item(), 'patient': loss_bag.item(), 'instance': loss_instance.item()})

        val_metrics = metric_logger.compute()
        metric_logger.reset()
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs} | "
                    f"Train Loss {train_metrics['loss_total']:.4f} | "
                    f"Val Loss {val_metrics['loss_total']:.4f} | "
                    f"Val Acc {val_metrics['accuracy']*100:.2f}%")

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
        acc = train_fold(config, fold_idx, train_loader, val_loader, class_names, logger)
        fold_accuracies.append(acc)
        logger.info(f"Fold {fold_idx+1} Accuracy: {acc*100:.2f}%")

    logger.info(f"\n5-Fold Average Accuracy: {np.mean(fold_accuracies)*100:.2f}%")

if __name__ == '__main__':
    main()

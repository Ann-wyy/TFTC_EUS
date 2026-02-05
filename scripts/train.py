#!/usr/bin/env python3
"""
训练脚本 - 多模态MIL肿瘤分类模型

用法:
    python scripts/train.py --data_root /path/to/data --epochs 100
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mil_classifier.models import MultiModalMILClassifier, create_model
from mil_classifier.data import (
    create_data_loaders,
    get_eus_transforms,
    get_wli_transforms
)
from mil_classifier.losses import create_loss_function
from mil_classifier.utils import MetricsCalculator, AverageMeter, EarlyStopping
from configs.config import Config, get_config


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()

    loss_meter = AverageMeter()
    patient_loss_meter = AverageMeter()
    frame_loss_meter = AverageMeter()
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        # 数据移到设备
        eus_frames = batch['eus_frames'].to(device)
        wli_frames = batch['wli_frames'].to(device)
        labels = batch['labels'].to(device)
        frame_labels = batch['frame_labels'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()

        # 前向传播
        if scaler is not None:
            with autocast():
                outputs = model(
                    eus_frames, wli_frames, masks,
                    return_attention=False
                )
                losses = criterion(
                    patient_logits=outputs['patient_logits'],
                    patient_labels=labels,
                    frame_logits=outputs.get('frame_logits'),
                    frame_labels=frame_labels,
                    mask=masks
                )

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                eus_frames, wli_frames, masks,
                return_attention=False
            )
            losses = criterion(
                patient_logits=outputs['patient_logits'],
                patient_labels=labels,
                frame_logits=outputs.get('frame_logits'),
                frame_labels=frame_labels,
                mask=masks
            )

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # 统计
        batch_size = eus_frames.size(0)
        loss_meter.update(losses['total'].item(), batch_size)
        patient_loss_meter.update(losses['patient'].item(), batch_size)
        frame_loss_meter.update(losses['frame'].item(), batch_size)

        preds = outputs['patient_logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size

        # 日志
        if logger and batch_idx % 20 == 0:
            logger.info(
                f'  Batch [{batch_idx}/{len(dataloader)}] '
                f'Loss: {loss_meter.avg:.4f} '
                f'Acc: {100.*correct/total:.2f}%'
            )

    return {
        'loss': loss_meter.avg,
        'patient_loss': patient_loss_meter.avg,
        'frame_loss': frame_loss_meter.avg,
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    num_classes: int,
    class_names
) -> Dict:
    """评估"""
    model.eval()

    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator(num_classes, class_names)

    for batch in dataloader:
        eus_frames = batch['eus_frames'].to(device)
        wli_frames = batch['wli_frames'].to(device)
        labels = batch['labels'].to(device)
        frame_labels = batch['frame_labels'].to(device)
        masks = batch['masks'].to(device)

        outputs = model(
            eus_frames, wli_frames, masks,
            return_attention=True
        )

        losses = criterion(
            patient_logits=outputs['patient_logits'],
            patient_labels=labels,
            frame_logits=outputs.get('frame_logits'),
            frame_labels=frame_labels,
            mask=masks
        )

        loss_meter.update(losses['total'].item(), eus_frames.size(0))

        probs = F.softmax(outputs['patient_logits'], dim=-1)
        preds = probs.argmax(dim=-1)
        metrics_calc.update(preds, labels, probs)

    metrics = metrics_calc.compute()
    metrics['loss'] = loss_meter.avg

    return metrics, metrics_calc


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    save_path: str
):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def train(config: Config):
    """主训练函数"""
    # 设置
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # 日志
    log_dir = Path(config.training.checkpoint_dir) / 'logs'
    logger = setup_logging(str(log_dir), config.experiment_name)
    logger.info(f"开始训练: {config.experiment_name}")
    logger.info(f"设备: {device}")

    # 数据增强
    transform_eus_train = get_eus_transforms(
        img_size=config.data.img_size,
        is_training=True
    )
    transform_wli_train = get_wli_transforms(
        img_size=config.data.img_size,
        is_training=True
    )
    transform_eus_val = get_eus_transforms(
        img_size=config.data.img_size,
        is_training=False
    )
    transform_wli_val = get_wli_transforms(
        img_size=config.data.img_size,
        is_training=False
    )

    # 数据加载器
    logger.info("加载数据...")
    loaders = create_data_loaders(
        data_root=config.data.data_root,
        train_csv=config.data.train_csv,
        val_csv=config.data.val_csv,
        test_csv=config.data.test_csv if hasattr(config.data, 'test_csv') else None,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        transform_eus_train=transform_eus_train,
        transform_wli_train=transform_wli_train,
        transform_eus_val=transform_eus_val,
        transform_wli_val=transform_wli_val,
        use_balanced_sampler=True
    )

    # 创建模型
    logger.info("创建模型...")
    model = create_model(config)
    model = model.to(device)
    logger.info(f"模型参数: {model.num_parameters:,}")
    logger.info(f"可训练参数: {model.num_trainable_parameters:,}")

    # 损失函数
    class_weights = None
    if hasattr(loaders['train'].dataset, 'get_class_weights'):
        class_weights = loaders['train'].dataset.get_class_weights().tolist()

    criterion = create_loss_function(
        loss_type=config.training.loss_type,
        num_classes=config.data.num_classes,
        class_weights=class_weights,
        gamma=config.training.focal_gamma,
        auxiliary_weight=config.classifier.auxiliary_weight,
        use_auxiliary=config.classifier.use_auxiliary_task
    )

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # 学习率调度器
    if config.training.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs - config.training.warmup_epochs,
            eta_min=config.training.min_lr
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=10,
            factor=0.5,
            min_lr=config.training.min_lr
        )

    # 混合精度
    scaler = GradScaler() if config.training.use_amp else None

    # 早停
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode='max'
    )

    # 训练循环
    best_f1 = 0.0
    checkpoint_dir = Path(config.training.checkpoint_dir) / config.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始训练循环...")

    for epoch in range(config.training.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
        logger.info("-" * 50)

        # 训练
        train_metrics = train_one_epoch(
            model=model,
            dataloader=loaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_grad_norm=config.training.max_grad_norm,
            logger=logger
        )

        # 验证
        val_metrics, val_calc = evaluate(
            model=model,
            dataloader=loaders['val'],
            criterion=criterion,
            device=device,
            num_classes=config.data.num_classes,
            class_names=config.data.class_names
        )

        # 更新学习率
        if config.training.scheduler == 'cosine':
            if epoch >= config.training.warmup_epochs:
                scheduler.step()
        else:
            scheduler.step(val_metrics['loss'])

        # 日志
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']*100:.2f}%")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}%")
        logger.info(f"Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        logger.info(f"Learning Rate: {current_lr:.2e}")

        # 保存最佳模型
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                save_path=str(checkpoint_dir / 'best_model.pth')
            )
            logger.info(f"  -> 保存最佳模型! F1: {best_f1:.4f}")

        # 早停检查
        if early_stopping(epoch, val_metrics['f1_macro']):
            logger.info(f"早停触发于 epoch {epoch+1}")
            break

    # 训练完成
    logger.info("\n训练完成!")
    logger.info(f"最佳验证 F1: {best_f1:.4f}")

    # 加载最佳模型进行测试
    if 'test' in loaders:
        logger.info("\n在测试集上评估...")
        checkpoint = torch.load(str(checkpoint_dir / 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics, test_calc = evaluate(
            model=model,
            dataloader=loaders['test'],
            criterion=criterion,
            device=device,
            num_classes=config.data.num_classes,
            class_names=config.data.class_names
        )

        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        logger.info(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info("\n分类报告:")
        logger.info(test_calc.get_classification_report())

        # 保存混淆矩阵
        test_calc.plot_confusion_matrix(
            save_path=str(checkpoint_dir / 'confusion_matrix.png')
        )

    logger.info(f"\n结果保存至: {checkpoint_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='训练多模态MIL肿瘤分类模型'
    )

    # 数据参数
    parser.add_argument('--data_root', type=str, default='data',
                        help='数据根目录')
    parser.add_argument('--train_csv', type=str, default='train.csv',
                        help='训练集CSV')
    parser.add_argument('--val_csv', type=str, default='val.csv',
                        help='验证集CSV')
    parser.add_argument('--test_csv', type=str, default=None,
                        help='测试集CSV')

    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet18', 'convnext_tiny', 'vit_b_16'],
                        help='Backbone架构')
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        choices=['concat', 'cross_attention', 'both'],
                        help='融合方式')
    parser.add_argument('--mil_pooling', type=str, default='gated_attention',
                        choices=['attention', 'gated_attention', 'transformer', 'max', 'mean'],
                        help='MIL池化方式')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['focal', 'ce', 'class_balanced'],
                        help='损失函数')

    # 其他
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点目录')
    parser.add_argument('--experiment_name', type=str, default='mil_experiment',
                        help='实验名称')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载进程数')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建配置
    config = get_config()

    # 更新配置
    config.data.data_root = args.data_root
    config.data.train_csv = args.train_csv
    config.data.val_csv = args.val_csv
    if args.test_csv:
        config.data.test_csv = args.test_csv

    config.encoder.backbone = args.backbone
    config.fusion.fusion_type = args.fusion_type
    config.mil.pooling_type = args.mil_pooling

    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.loss_type = args.loss
    config.training.checkpoint_dir = args.checkpoint_dir
    config.training.num_workers = args.num_workers

    config.experiment_name = args.experiment_name
    config.seed = args.seed
    config.device = args.device

    # 训练
    train(config)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main()

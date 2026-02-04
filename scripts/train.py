#!/usr/bin/env python3
"""
Training Script for TFTC Model.

This script trains the Trimodal Fusion Transformer Classifier for
submucosal tumor classification.

Usage:
    python scripts/train.py --config configs/default_config.py
    python scripts/train.py --data_root /path/to/data --epochs 100
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tftc.models import TFTC, TFTCAblation, create_tftc_from_dataclass
from tftc.data import create_data_loaders
from tftc.losses import FocalLoss, create_loss_function
from tftc.utils import MetricsCalculator, EarlyStopping, AverageMeter
from configs.default_config import TFTCConfig, get_default_config, get_ablation_config


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging to file and console."""
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
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(
    model: nn.Module,
    config: TFTCConfig
) -> torch.optim.Optimizer:
    """Create optimizer with optional layer-wise learning rate decay."""
    # Separate parameters for different learning rates
    encoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'wli_encoder' in name or 'eus_encoder' in name:
            encoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': encoder_params, 'lr': config.training.learning_rate * 0.1},
        {'params': other_params, 'lr': config.training.learning_rate}
    ]

    if config.training.optimizer == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.training.betas
        )
    else:
        # RAdam
        from torch.optim import RAdam
        optimizer = RAdam(
            param_groups,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.training.betas
        )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TFTCConfig,
    num_training_steps: int
):
    """Create learning rate scheduler."""
    if config.training.scheduler == 'cosine':
        # Warmup + Cosine Annealing
        warmup_steps = config.training.warmup_epochs
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs - warmup_steps,
            eta_min=config.training.min_lr
        )
    else:
        # Reduce on Plateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.training.plateau_patience,
            factor=config.training.plateau_factor,
            min_lr=config.training.min_lr
        )

    return scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()

    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        wli_images = batch['wli_image'].to(device)
        eus_images = batch['eus_image'].to(device)
        locations = batch['location'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(
                    wli_image=wli_images,
                    eus_image=eus_images,
                    location=locations
                )
                loss = criterion(outputs, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                wli_image=wli_images,
                eus_image=eus_images,
                location=locations
            )
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # Statistics
        loss_meter.update(loss.item(), wli_images.size(0))

        # Handle soft labels from MixUp
        if labels.dim() == 2:
            labels = labels.argmax(dim=1)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Log progress
        if logger and batch_idx % 50 == 0:
            logger.info(
                f'  Batch [{batch_idx}/{len(dataloader)}] '
                f'Loss: {loss_meter.avg:.4f} Acc: {100.*correct/total:.2f}%'
            )

    accuracy = correct / total
    return loss_meter.avg, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: List[str]
) -> Tuple[float, Dict]:
    """
    Evaluate model on validation/test set.

    Returns:
        Tuple of (average_loss, metrics_dict).
    """
    model.eval()

    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator(num_classes, class_names)

    for batch in dataloader:
        wli_images = batch['wli_image'].to(device)
        eus_images = batch['eus_image'].to(device)
        locations = batch['location'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            wli_image=wli_images,
            eus_image=eus_images,
            location=locations
        )

        # Handle soft labels
        if labels.dim() == 2:
            labels = labels.argmax(dim=1)

        loss = criterion(outputs, labels)
        loss_meter.update(loss.item(), wli_images.size(0))

        # Get predictions and probabilities
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        metrics_calc.update(preds, labels, probs)

    metrics = metrics_calc.compute()
    metrics['loss'] = loss_meter.avg

    return loss_meter.avg, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    save_path: str,
    config: TFTCConfig
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }
    torch.save(checkpoint, save_path)


def train(
    config: TFTCConfig,
    active_modalities: Optional[List[str]] = None
):
    """
    Main training function.

    Args:
        config: Training configuration.
        active_modalities: List of active modalities for ablation.
    """
    # Setup
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # Logging
    log_dir = Path(config.training.checkpoint_dir) / 'logs'
    logger = setup_logging(str(log_dir), config.experiment_name)
    logger.info(f"Starting training: {config.experiment_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Active modalities: {active_modalities or ['wli', 'eus', 'location']}")

    # Create data loaders
    logger.info("Loading data...")
    loaders = create_data_loaders(
        data_root=config.data.data_root,
        train_csv=config.data.train_csv,
        val_csv=config.data.val_csv,
        test_csv=config.data.test_csv if hasattr(config.data, 'test_csv') else None,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        img_size=config.image.wli_size,
        location_categories=config.data.location_categories,
        tumor_classes=config.data.tumor_classes,
        use_weighted_sampler=True
    )

    # Create model
    logger.info("Creating model...")
    if active_modalities:
        model = TFTC(
            wli_encoder_config={
                'model_name': config.wli_encoder.model_name,
                'pretrained': config.wli_encoder.pretrained,
                'output_dim': config.wli_encoder.output_dim,
                'freeze_layers': config.wli_encoder.freeze_layers
            },
            eus_encoder_config={
                'model_name': config.eus_encoder.model_name,
                'pretrained': config.eus_encoder.pretrained,
                'output_dim': config.eus_encoder.output_dim,
                'freeze_layers': config.eus_encoder.freeze_layers
            },
            location_config={
                'num_locations': config.location_embedder.num_locations,
                'embedding_dim': config.location_embedder.embedding_dim,
                'hidden_dim': config.location_embedder.hidden_dim,
                'use_mlp': config.location_embedder.use_mlp
            },
            fusion_config={
                'num_layers': config.fusion_transformer.num_layers,
                'num_heads': config.fusion_transformer.num_heads,
                'ff_dim': config.fusion_transformer.ff_dim,
                'dropout': config.fusion_transformer.dropout
            },
            classifier_config={
                'hidden_dims': config.classifier.hidden_dims,
                'num_classes': config.classifier.num_classes,
                'dropout': config.classifier.dropout
            },
            active_modalities=active_modalities
        )
    else:
        model = create_tftc_from_dataclass(config)

    model = model.to(device)
    logger.info(f"Model parameters: {model.num_parameters:,}")
    logger.info(f"Trainable parameters: {model.num_trainable_parameters:,}")

    # Create loss function
    class_weights = None
    if config.training.focal_alpha:
        class_weights = config.training.focal_alpha
    elif hasattr(loaders['train'].dataset, 'get_class_weights'):
        class_weights = loaders['train'].dataset.get_class_weights().tolist()

    criterion = create_loss_function(
        loss_type=config.training.loss_fn,
        num_classes=config.classifier.num_classes,
        class_weights=class_weights,
        gamma=config.training.focal_gamma
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(
        optimizer, config,
        num_training_steps=len(loaders['train']) * config.training.num_epochs
    )

    # Mixed precision training
    scaler = GradScaler() if config.training.use_amp else None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode='max',
        restore_best=True
    )

    # Training loop
    best_f1 = 0.0
    checkpoint_dir = Path(config.training.checkpoint_dir) / config.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training loop...")

    for epoch in range(config.training.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
        logger.info("-" * 50)

        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=loaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_grad_norm=config.training.max_grad_norm,
            logger=logger
        )

        # Evaluate
        val_loss, val_metrics = evaluate(
            model=model,
            dataloader=loaders['val'],
            criterion=criterion,
            device=device,
            num_classes=config.classifier.num_classes,
            class_names=config.data.tumor_classes
        )

        # Update scheduler
        if config.training.scheduler == 'cosine':
            if epoch >= config.training.warmup_epochs:
                scheduler.step()
        else:
            scheduler.step(val_loss)

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}%")
        logger.info(f"Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        logger.info(f"Learning Rate: {current_lr:.2e}")

        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                save_path=str(checkpoint_dir / 'best_model.pth'),
                config=config
            )
            logger.info(f"  -> New best model saved! F1: {best_f1:.4f}")

        # Check early stopping
        if early_stopping(epoch, val_metrics['f1_macro'], model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best model and final evaluation
    logger.info("\nTraining completed!")
    logger.info(f"Best validation F1: {best_f1:.4f}")

    early_stopping.restore(model)

    # Test evaluation (if test set available)
    if 'test' in loaders:
        logger.info("\nEvaluating on test set...")
        test_loss, test_metrics = evaluate(
            model=model,
            dataloader=loaders['test'],
            criterion=criterion,
            device=device,
            num_classes=config.classifier.num_classes,
            class_names=config.data.tumor_classes
        )

        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        logger.info(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")

        # Print classification report
        metrics_calc = MetricsCalculator(
            config.classifier.num_classes,
            config.data.tumor_classes
        )
        # Re-run to populate metrics calculator
        for batch in loaders['test']:
            wli_images = batch['wli_image'].to(device)
            eus_images = batch['eus_image'].to(device)
            locations = batch['location'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                outputs = model(
                    wli_image=wli_images,
                    eus_image=eus_images,
                    location=locations
                )

            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            metrics_calc.update(preds, labels, probs)

        logger.info("\nClassification Report:")
        logger.info(metrics_calc.get_classification_report())

        # Save confusion matrix plot
        fig = metrics_calc.plot_confusion_matrix(
            save_path=str(checkpoint_dir / 'confusion_matrix.png')
        )
        plt.close(fig)

        # Save ROC curves
        fig = metrics_calc.plot_roc_curves(
            save_path=str(checkpoint_dir / 'roc_curves.png')
        )
        if fig:
            plt.close(fig)

    logger.info(f"\nResults saved to: {checkpoint_dir}")


def run_ablation_study(config: TFTCConfig):
    """
    Run ablation study to evaluate contribution of each modality.
    """
    ablation_configs = [
        (['wli'], 'WLI Only'),
        (['eus'], 'EUS Only'),
        (['wli', 'eus'], 'WLI + EUS'),
        (['wli', 'location'], 'WLI + Location'),
        (['eus', 'location'], 'EUS + Location'),
        (['wli', 'eus', 'location'], 'Full Model (WLI + EUS + Location)')
    ]

    results = []

    for modalities, name in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Running ablation: {name}")
        print(f"{'='*60}")

        ablation_config = get_ablation_config(modalities)
        ablation_config.experiment_name = f"ablation_{name.replace(' ', '_').lower()}"

        train(ablation_config, active_modalities=modalities)

        # Note: In practice, you would collect and compare results here


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TFTC model for SMT classification'
    )

    # Data arguments
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory containing the data')
    parser.add_argument('--train_csv', type=str, default='train.csv',
                        help='Training CSV file')
    parser.add_argument('--val_csv', type=str, default='val.csv',
                        help='Validation CSV file')
    parser.add_argument('--test_csv', type=str, default=None,
                        help='Test CSV file (optional)')

    # Model arguments
    parser.add_argument('--wli_encoder', type=str, default='swin_v2_b',
                        choices=['swin_v2_b', 'swin_v2_s', 'vit_l_16', 'vit_b_16'],
                        help='WLI encoder architecture')
    parser.add_argument('--eus_encoder', type=str, default='convnext_base',
                        choices=['convnext_base', 'convnext_small', 'resnet50', 'resnet101'],
                        help='EUS encoder architecture')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['focal', 'weighted_ce', 'ce'],
                        help='Loss function')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')

    # Other arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--experiment_name', type=str, default='tftc_experiment',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Ablation study
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--modalities', type=str, nargs='+',
                        default=['wli', 'eus', 'location'],
                        help='Active modalities')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config
    config = get_default_config()

    # Update config with command line arguments
    config.data.data_root = args.data_root
    config.data.train_csv = args.train_csv
    config.data.val_csv = args.val_csv
    if args.test_csv:
        config.data.test_csv = args.test_csv

    config.wli_encoder.model_name = args.wli_encoder
    config.eus_encoder.model_name = args.eus_encoder

    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.loss_fn = args.loss
    config.training.focal_gamma = args.focal_gamma
    config.training.checkpoint_dir = args.checkpoint_dir
    config.training.num_workers = args.num_workers

    config.experiment_name = args.experiment_name
    config.seed = args.seed
    config.device = args.device

    # Run training or ablation
    if args.ablation:
        run_ablation_study(config)
    else:
        active_modalities = args.modalities if args.modalities != ['wli', 'eus', 'location'] else None
        train(config, active_modalities=active_modalities)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    main()

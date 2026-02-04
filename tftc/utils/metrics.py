"""
Evaluation Metrics Module.

This module implements comprehensive evaluation metrics for
SMT classification, including accuracy, precision, recall, F1-score,
confusion matrix, and ROC-AUC.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """
    Comprehensive metrics calculator for multi-class classification.

    Computes:
    - Accuracy
    - Precision (macro, micro, weighted, per-class)
    - Recall (macro, micro, weighted, per-class)
    - F1-Score (macro, micro, weighted, per-class)
    - Confusion Matrix
    - ROC Curves and AUC (per-class and macro)

    Args:
        num_classes: Number of classes.
        class_names: List of class names for reporting.
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]

        # Storage for predictions and labels
        self.reset()

    def reset(self):
        """Reset accumulated predictions and labels."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        probs: Optional[torch.Tensor] = None
    ):
        """
        Update with batch predictions and labels.

        Args:
            preds: Predicted class indices of shape (B,).
            labels: Ground truth labels of shape (B,).
            probs: Class probabilities of shape (B, num_classes).
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.tolist())
        if probs is not None:
            self.all_probs.extend(probs.tolist())

    def compute(self) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """
        Compute all metrics.

        Returns:
            Dictionary containing all computed metrics.
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs) if self.all_probs else None

        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(labels, preds)

        # Precision, Recall, F1 (different averages)
        for avg in ['macro', 'micro', 'weighted']:
            metrics[f'precision_{avg}'] = precision_score(
                labels, preds, average=avg, zero_division=0
            )
            metrics[f'recall_{avg}'] = recall_score(
                labels, preds, average=avg, zero_division=0
            )
            metrics[f'f1_{avg}'] = f1_score(
                labels, preds, average=avg, zero_division=0
            )

        # Per-class metrics
        metrics['precision_per_class'] = precision_score(
            labels, preds, average=None, zero_division=0
        )
        metrics['recall_per_class'] = recall_score(
            labels, preds, average=None, zero_division=0
        )
        metrics['f1_per_class'] = f1_score(
            labels, preds, average=None, zero_division=0
        )

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(
            labels, preds, labels=list(range(self.num_classes))
        )

        # ROC-AUC (if probabilities available)
        if probs is not None and len(probs) > 0:
            try:
                # One-hot encode labels for multi-class ROC
                labels_onehot = np.eye(self.num_classes)[labels]

                # Per-class AUC
                auc_per_class = []
                for i in range(self.num_classes):
                    if len(np.unique(labels_onehot[:, i])) > 1:
                        auc_i = roc_auc_score(labels_onehot[:, i], probs[:, i])
                    else:
                        auc_i = 0.0
                    auc_per_class.append(auc_i)

                metrics['auc_per_class'] = np.array(auc_per_class)
                metrics['auc_macro'] = np.mean(auc_per_class)

                # Macro AUC using sklearn (handles edge cases)
                try:
                    metrics['auc_macro_sklearn'] = roc_auc_score(
                        labels, probs, multi_class='ovr', average='macro'
                    )
                except ValueError:
                    metrics['auc_macro_sklearn'] = 0.0

            except Exception as e:
                metrics['auc_error'] = str(e)
                metrics['auc_per_class'] = np.zeros(self.num_classes)
                metrics['auc_macro'] = 0.0

        return metrics

    def get_classification_report(self) -> str:
        """
        Get detailed classification report as string.

        Returns:
            Classification report string.
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        return classification_report(
            labels, preds,
            target_names=self.class_names,
            zero_division=0
        )

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        return confusion_matrix(labels, preds, labels=list(range(self.num_classes)))

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            save_path: Path to save the figure.
            normalize: Whether to normalize by row (true labels).
            figsize: Figure size.

        Returns:
            Matplotlib figure.
        """
        cm = self.get_confusion_matrix()

        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_roc_curves(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Optional[plt.Figure]:
        """
        Plot ROC curves for each class.

        Args:
            save_path: Path to save the figure.
            figsize: Figure size.

        Returns:
            Matplotlib figure or None if probabilities not available.
        """
        if not self.all_probs:
            return None

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        labels_onehot = np.eye(self.num_classes)[labels]

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, self.num_classes))

        for i, (name, color) in enumerate(zip(self.class_names, colors)):
            if len(np.unique(labels_onehot[:, i])) > 1:
                fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(
                    fpr, tpr,
                    color=color,
                    label=f'{name} (AUC = {roc_auc:.3f})'
                )

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (One-vs-Rest)')
        ax.legend(loc='lower right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def summary(self) -> str:
        """
        Get summary string of key metrics.

        Returns:
            Summary string.
        """
        metrics = self.compute()

        lines = [
            "=" * 50,
            "EVALUATION METRICS SUMMARY",
            "=" * 50,
            f"Accuracy: {metrics['accuracy']:.4f}",
            "",
            "F1-Scores:",
            f"  Macro:    {metrics['f1_macro']:.4f}",
            f"  Micro:    {metrics['f1_micro']:.4f}",
            f"  Weighted: {metrics['f1_weighted']:.4f}",
            "",
            "Per-Class Metrics:",
        ]

        for i, name in enumerate(self.class_names):
            lines.append(
                f"  {name:25s} | "
                f"P: {metrics['precision_per_class'][i]:.3f} | "
                f"R: {metrics['recall_per_class'][i]:.3f} | "
                f"F1: {metrics['f1_per_class'][i]:.3f}"
            )

        if 'auc_macro' in metrics:
            lines.extend([
                "",
                f"AUC (Macro): {metrics['auc_macro']:.4f}"
            ])

        lines.append("=" * 50)

        return '\n'.join(lines)


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss, 'max' for metrics like accuracy.
        restore_best: Whether to restore best model weights when stopped.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.should_stop = False

    def __call__(
        self,
        epoch: int,
        score: float,
        model: Optional[torch.nn.Module] = None
    ) -> bool:
        """
        Check if training should stop.

        Args:
            epoch: Current epoch number.
            score: Current validation score.
            model: Model to save best weights from.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self._update_best(epoch, score, model)
            return False

        is_better = self._is_better(score)

        if is_better:
            self._update_best(epoch, score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def _is_better(self, score: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def _update_best(
        self,
        epoch: int,
        score: float,
        model: Optional[torch.nn.Module]
    ):
        """Update best score and optionally save model weights."""
        self.best_score = score
        self.best_epoch = epoch
        if model is not None and self.restore_best:
            self.best_weights = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    def restore(self, model: torch.nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking loss during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update with new value.

        Args:
            val: New value.
            n: Number of samples (for batch averaging).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(
    preds: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_classes: int = 5,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Convenience function to compute all metrics at once.

    Args:
        preds: Predicted class indices.
        labels: Ground truth labels.
        probs: Class probabilities.
        num_classes: Number of classes.
        class_names: Class names for reporting.

    Returns:
        Dictionary of metrics.
    """
    calculator = MetricsCalculator(num_classes, class_names)
    calculator.update(preds, labels, probs)
    return calculator.compute()

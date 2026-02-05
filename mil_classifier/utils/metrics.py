"""
评估指标模块

包含:
- 准确率、精确率、召回率、F1
- AUROC (每类)
- 混淆矩阵
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    指标计算器
    """

    def __init__(
        self,
        num_classes: int = 6,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [
            "平滑肌瘤", "脂肪瘤", "间质瘤",
            "神经内分泌瘤", "异位胰腺", "其他"
        ]

        self.reset()

    def reset(self):
        """重置"""
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
        更新

        Args:
            preds: 预测类别 [B]
            labels: 真实标签 [B]
            probs: 类别概率 [B, C]
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

    def compute(self) -> Dict:
        """
        计算所有指标

        Returns:
            指标字典
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs) if self.all_probs else None

        metrics = {}

        # 准确率
        metrics['accuracy'] = accuracy_score(labels, preds)

        # 精确率、召回率、F1
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

        # 每类指标
        metrics['precision_per_class'] = precision_score(
            labels, preds, average=None, zero_division=0
        )
        metrics['recall_per_class'] = recall_score(
            labels, preds, average=None, zero_division=0
        )
        metrics['f1_per_class'] = f1_score(
            labels, preds, average=None, zero_division=0
        )

        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(
            labels, preds, labels=list(range(self.num_classes))
        )

        # AUROC
        if probs is not None and len(probs) > 0:
            try:
                labels_onehot = np.eye(self.num_classes)[labels]

                auc_per_class = []
                for i in range(self.num_classes):
                    if len(np.unique(labels_onehot[:, i])) > 1:
                        auc_i = roc_auc_score(labels_onehot[:, i], probs[:, i])
                    else:
                        auc_i = 0.0
                    auc_per_class.append(auc_i)

                metrics['auc_per_class'] = np.array(auc_per_class)
                metrics['auc_macro'] = np.mean(auc_per_class)

            except Exception as e:
                metrics['auc_error'] = str(e)

        return metrics

    def get_classification_report(self) -> str:
        """获取分类报告"""
        return classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            zero_division=0
        )

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """绘制混淆矩阵"""
        cm = confusion_matrix(
            self.all_labels,
            self.all_preds,
            labels=list(range(self.num_classes))
        )

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
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        ax.set_title('混淆矩阵')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def summary(self) -> str:
        """生成摘要"""
        metrics = self.compute()

        lines = [
            "=" * 50,
            "评估指标摘要",
            "=" * 50,
            f"准确率: {metrics['accuracy']:.4f}",
            f"F1 (Macro): {metrics['f1_macro']:.4f}",
            f"F1 (Weighted): {metrics['f1_weighted']:.4f}",
            "",
            "各类别指标:",
        ]

        for i, name in enumerate(self.class_names):
            lines.append(
                f"  {name:12s} | "
                f"P: {metrics['precision_per_class'][i]:.3f} | "
                f"R: {metrics['recall_per_class'][i]:.3f} | "
                f"F1: {metrics['f1_per_class'][i]:.3f}"
            )

        if 'auc_macro' in metrics:
            lines.append(f"\nAUROC (Macro): {metrics['auc_macro']:.4f}")

        lines.append("=" * 50)

        return '\n'.join(lines)


class AverageMeter:
    """平均值计算器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停"""

    def __init__(
        self,
        patience: int = 10,
        mode: str = 'max',
        min_delta: float = 0.0
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, epoch: int, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

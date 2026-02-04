"""
Focal Loss and Other Loss Functions.

This module implements various loss functions for handling class imbalance
in SMT classification, including Focal Loss and Weighted Cross-Entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss down-weights well-classified examples and focuses on
    hard, misclassified examples. This is particularly useful when
    some tumor types (e.g., gastric leiomyoma) have very few samples.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Class weights. Can be:
            - None: No class weighting
            - float: Same weight for all classes
            - List[float]: Per-class weights
        gamma: Focusing parameter (default: 2.0).
            - γ = 0: Equivalent to cross-entropy
            - γ > 0: Reduces the relative loss for well-classified examples
        reduction: Reduction method ('none', 'mean', 'sum').
        label_smoothing: Label smoothing factor (default: 0.0).
    """

    def __init__(
        self,
        alpha: Optional[float | List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Handle alpha (class weights)
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha])
        else:
            self.alpha = torch.tensor(alpha)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions (logits) of shape (B, C) where C is num_classes.
            targets: Ground truth labels of shape (B,) with class indices
                     OR shape (B, C) with soft labels/one-hot encoding.

        Returns:
            Focal loss value.
        """
        # Handle soft labels (from MixUp/CutMix)
        if targets.dim() == 2:
            return self._forward_soft_labels(inputs, targets)

        num_classes = inputs.shape[1]

        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
            return self._forward_soft_labels(inputs, smooth_targets)

        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)

        # Get probability of target class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha (class weights)
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            if alpha.numel() == 1:
                alpha_t = alpha.expand(num_classes)
            else:
                alpha_t = alpha

            alpha_t = alpha_t.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def _forward_soft_labels(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss with soft labels.

        Args:
            inputs: Predictions (logits) of shape (B, C).
            targets: Soft labels of shape (B, C).

        Returns:
            Focal loss value.
        """
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)

        # Compute focal weight per class
        focal_weight = (1 - probs) ** self.gamma

        # Compute cross-entropy per class
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -targets * log_probs

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha (class weights)
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            focal_loss = alpha.unsqueeze(0) * focal_loss

        # Sum over classes and apply reduction
        focal_loss = focal_loss.sum(dim=1)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for class imbalance.

    Applies different weights to each class based on their frequency.

    Args:
        weight: Class weights tensor of shape (num_classes,).
        label_smoothing: Label smoothing factor.
        reduction: Reduction method.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            inputs: Predictions (logits) of shape (B, C).
            targets: Ground truth labels of shape (B,) or (B, C) for soft labels.

        Returns:
            Loss value.
        """
        weight = self.weight.to(inputs.device) if self.weight is not None else None

        # Handle soft labels
        if targets.dim() == 2:
            log_probs = F.log_softmax(inputs, dim=1)
            loss = -targets * log_probs
            if weight is not None:
                loss = weight.unsqueeze(0) * loss
            loss = loss.sum(dim=1)
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss

        return F.cross_entropy(
            inputs,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.

    Label smoothing prevents the model from becoming overconfident
    and improves generalization.

    Args:
        num_classes: Number of classes.
        smoothing: Smoothing factor (0.0 to 1.0).
        reduction: Reduction method.
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            inputs: Predictions (logits) of shape (B, C).
            targets: Ground truth labels of shape (B,).

        Returns:
            Loss value.
        """
        # Handle soft labels
        if targets.dim() == 2:
            smooth_targets = targets
        else:
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines multiple losses with weights.

    Useful for multi-task learning or combining different loss objectives.

    Args:
        losses: List of loss functions.
        weights: Weights for each loss (default: equal weights).
    """

    def __init__(
        self,
        losses: List[nn.Module],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            inputs: Predictions (logits) of shape (B, C).
            targets: Ground truth labels.

        Returns:
            Combined loss value.
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss = total_loss + weight * loss_fn(inputs, targets)
        return total_loss


def create_loss_function(
    loss_type: str = 'focal',
    num_classes: int = 5,
    class_weights: Optional[List[float]] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: Type of loss ('focal', 'weighted_ce', 'label_smoothing', 'ce').
        num_classes: Number of classes.
        class_weights: Class weights for imbalanced data.
        gamma: Focal loss gamma parameter.
        label_smoothing: Label smoothing factor.

    Returns:
        Loss function module.
    """
    if loss_type == 'focal':
        return FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing
        )
    elif loss_type == 'weighted_ce':
        weight = torch.tensor(class_weights) if class_weights else None
        return WeightedCrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing
        )
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=label_smoothing
        )
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

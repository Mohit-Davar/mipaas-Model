"""
losses.py
=============================================================================
Standardized Loss Functions for CheXpert Plus Multi-Label Training.
=============================================================================
Supported objectives:
1. MaskedBCEWithLogits: Standard Stanford CheXpert competition loss.
   Computes binary cross entropy per label with optional boolean masking
   to ignore uncertain annotations.
2. AsymmetricLoss (ASL): Multi-label loss addressing extreme negative-class
   prevalence with smooth probability margin clamping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _masked_mean(loss: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes masked mean across valid entries.
    If mask is None, returns standard tensor mean.
    """
    if mask is None:
        return loss.mean()
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class MaskedBCEWithLogits(nn.Module):
    """
    Binary Cross-Entropy with Logits supporting optional per-element masking.
    Standardized loss function for the CheXpert benchmark.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits.float(),
            targets.float(),
            pos_weight=self.pos_weight,
            reduction="none",
        )
        return _masked_mean(loss, mask)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ridnik et al.):
    Applies asymmetric focal focusing to separate easy negative examples
    from rare positive findings.
    Note: clip=0.0 and gamma_neg=1.0 ensures negative gradients are NEVER zeroed out.
    """

    def __init__(
        self,
        gamma_neg: float = 1.0,
        gamma_pos: float = 0.0,
        clip: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        
        p = torch.sigmoid(logits)
        p_neg = 1.0 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)
            
        log_pos = targets * torch.log(p.clamp(min=self.eps))
        log_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps))
        loss = log_pos + log_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = p * targets + p_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)

        return _masked_mean(-loss, mask)


def compute_pos_weight(
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_weight: float = 8.0,
) -> torch.Tensor:
    """
    Computes inverse frequency positive-class weight vector.
    """
    labels = labels.float()
    m = torch.ones_like(labels) if mask is None else mask.float()
    pos = (labels * m).sum(0).clamp(min=1.0)
    neg = ((1.0 - labels) * m).sum(0).clamp(min=1.0)
    return (neg / pos).clamp(min=1.0, max=max_weight)


def build_loss(name: str = "bce", pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory function for loss modules.
    """
    name_clean = name.strip().lower()
    if name_clean == "bce":
        return MaskedBCEWithLogits(pos_weight=pos_weight)
    elif name_clean == "asl":
        if pos_weight is not None:
            print("[loss] pos_weight is not used for ASL.")
        return AsymmetricLoss(gamma_neg=1.0, gamma_pos=0.0, clip=0.0)
    raise ValueError(f"Unknown loss function '{name}'. Choose 'bce' or 'asl'.")


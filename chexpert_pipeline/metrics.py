"""
metrics.py
=============================================================================
Standardized Evaluation Metrics and Analysis for CheXpert Plus.
=============================================================================
Features:
- Official Stanford CheXpert competition ROC-AUC computation (5 key pathologies).
- Comprehensive 14-pathology evaluation with support counts.
- Collapse detection: Flags static prediction collapse (pred_std < 0.02).
- Youden's J threshold optimization for clinical operating point calibration.
- 100% crash-free ASCII table formatting for Windows console compatibility.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from config import ALL_LABELS, COMPETITION_PATHOLOGIES


def _nanmean(values: List[float]) -> float:
    """Computes arithmetic mean ignoring NaN entries."""
    valid = [v for v in values if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_chexpert_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    labels: List[str] = ALL_LABELS,
) -> Dict[str, Any]:
    """
    Computes per-pathology ROC-AUC and official CheXpert competition metrics.
    """
    per_class: Dict[str, float] = {}
    support: Dict[str, Tuple[int, int]] = {}

    for c, name in enumerate(labels):
        if valid_mask is None:
            yt = y_true[:, c]
            yp = y_pred[:, c]
        else:
            m = valid_mask[:, c].astype(bool)
            yt = y_true[m, c]
            yp = y_pred[m, c]

        n_pos = int((yt == 1).sum())
        n_neg = int((yt == 0).sum())
        support[name] = (n_pos, n_neg)

        if n_pos > 0 and n_neg > 0:
            try:
                per_class[name] = float(roc_auc_score(yt, yp))
            except Exception:
                per_class[name] = float("nan")
        else:
            per_class[name] = float("nan")

    # Standard deviation of predictions across images (averaged over classes)
    pred_std = float(np.mean(np.std(y_pred, axis=0)))

    comp_aucs = [per_class[n] for n in COMPETITION_PATHOLOGIES if n in per_class]
    all_aucs = list(per_class.values())

    return {
        "per_class_auc": per_class,
        "per_class_support": support,
        "competition_mean_auc": _nanmean(comp_aucs),
        "mean_auc_all": _nanmean(all_aucs),
        "pred_std": pred_std,
        "collapsed": (pred_std < 0.02),
    }


def predictions_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertain: np.ndarray,
    labels: List[str] = ALL_LABELS,
) -> pd.DataFrame:
    """Standard prediction export (pred_*, true_*, unc_* columns) shared by train/evaluate/ensemble."""
    df = pd.DataFrame(y_pred, columns=[f"pred_{lbl}" for lbl in labels])
    for i, lbl in enumerate(labels):
        df[f"true_{lbl}"] = y_true[:, i]
        df[f"unc_{lbl}"] = uncertain[:, i].astype(bool)
    return df


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ALL_LABELS,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculates the optimal operating threshold per pathology using Youden's J statistic:
        J = Sensitivity + Specificity - 1 = TruePositiveRate - FalsePositiveRate
    Only entries with valid_mask=True (certain labels) are used when a mask is given.
    """
    thresholds: Dict[str, float] = {}
    for idx, label in enumerate(labels):
        m = slice(None) if valid_mask is None else valid_mask[:, idx].astype(bool)
        yt = y_true[m, idx]
        yp = y_pred[m, idx]
        if len(np.unique(yt)) > 1:
            fpr, tpr, thresh = roc_curve(yt, yp)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            thresholds[label] = float(np.clip(thresh[best_idx], 0.05, 0.95))
        else:
            thresholds[label] = 0.5
    return thresholds


def print_auc_summary(metrics: Dict[str, Any], title: str = "AUC Summary") -> None:
    """
    Prints a formatted evaluation table with clean ASCII borders.
    """
    print(f"\n  +-- {title}")
    for name, auc in metrics["per_class_auc"].items():
        n_pos, n_neg = metrics["per_class_support"][name]
        is_comp = "*" if name in COMPETITION_PATHOLOGIES else " "
        auc_str = "   nan" if math.isnan(auc) else f"{auc:.4f}"
        print(f"  | {is_comp} {name:<28s} {auc_str}   (pos={n_pos:>5,d}, neg={n_neg:>6,d})")
    
    print("  +------------------------------------------------------------------")
    print(f"  | Competition Mean AUC (5 key benchmark classes): {metrics['competition_mean_auc']:.4f}")
    if "competition_mean_auc_mapped" in metrics:
        print(f"  | Competition Mean AUC (policy-mapped labels):     {metrics['competition_mean_auc_mapped']:.4f}")
    print(f"  | Overall Mean AUC (all {len(metrics['per_class_auc'])} target classes):       {metrics['mean_auc_all']:.4f}")
    print(f"  | Prediction Dispersion (mean std across images): {metrics['pred_std']:.4f}")
    print("  +------------------------------------------------------------------")

    if metrics.get("collapsed", False):
        print("  !! WARNING: Model predictions are virtually constant across images (model collapsed).")
        print("  !! Check learning rates, output bias initialization, and ensure image features are discriminative.")


"""
ensemble.py
=============================================================================
Multi-Model Ensembling for CheXpert Plus.
=============================================================================
Combines prediction CSVs written by train.py / evaluate.py (pred_*, true_*, unc_*
columns) for the SAME split, and scores the ensemble on certain labels, exactly
like the per-model metrics.

Blending strategies:
  * mean     - simple probability average
  * weighted - weighted probability average (--weights)
  * rank     - normalized rank averaging per pathology

Example (after training three backbones):
  python ensemble.py --split valid --pred_files ^
      ..\\outputs\\densenet121_valid_predictions.csv ^
      ..\\outputs\\tf_efficientnetv2_s_valid_predictions.csv ^
      ..\\outputs\\convnext_small_valid_predictions.csv
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import ALL_LABELS, OUTPUT_DIR
from metrics import compute_chexpert_auc, print_auc_summary, predictions_frame

PRED_COLS = [f"pred_{lbl}" for lbl in ALL_LABELS]
TRUE_COLS = [f"true_{lbl}" for lbl in ALL_LABELS]
UNC_COLS = [f"unc_{lbl}" for lbl in ALL_LABELS]


def parse_args():
    parser = argparse.ArgumentParser(description="CheXpert Plus Ensemble Engine")
    parser.add_argument("--pred_files", nargs="+", required=True, help="Prediction CSVs (same split) to ensemble")
    parser.add_argument("--method", type=str, default="mean", choices=["mean", "weighted", "rank"], help="Ensemble method")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Weights for --method weighted")
    parser.add_argument("--split", type=str, default="valid", help="Split name used for the output file name")
    parser.add_argument("--out_dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    return parser.parse_args()


def rank_average(pred_matrices: List[np.ndarray]) -> np.ndarray:
    """Computes normalized rank average across models for each pathology column."""
    n_models = len(pred_matrices)
    n_samples, n_classes = pred_matrices[0].shape
    ensemble_ranks = np.zeros((n_samples, n_classes), dtype=np.float32)
    for c in range(n_classes):
        for m in range(n_models):
            ensemble_ranks[:, c] += (rankdata(pred_matrices[m][:, c]) / n_samples) / n_models
    return ensemble_ranks


def run_ensemble():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(" CheXpert Plus Ensemble Engine")
    print(f" Models Ensembled  : {len(args.pred_files)}")
    print(f" Blending Strategy : {args.method.upper()}")
    print("=" * 68)

    frames = []
    for f in args.pred_files:
        df_p = pd.read_csv(f)
        missing = [c for c in PRED_COLS + TRUE_COLS + UNC_COLS if c not in df_p.columns]
        if missing:
            raise SystemExit(f"{f} lacks columns {missing[:3]}... Re-export it with train.py or evaluate.py.")
        frames.append(df_p)

    ref = frames[0]
    for f, df_p in zip(args.pred_files[1:], frames[1:]):
        if len(df_p) != len(ref) or not np.array_equal(df_p[TRUE_COLS].values, ref[TRUE_COLS].values):
            raise SystemExit(f"{f} does not cover the same images in the same order as {args.pred_files[0]}. "
                             f"Evaluate all models on the same split without --sample_limit.")

    y_true = ref[TRUE_COLS].values.astype(np.float32)
    unc = ref[UNC_COLS].values.astype(bool)
    pred_matrices = [df_p[PRED_COLS].values.astype(np.float32) for df_p in frames]

    print("\n  Individual models (competition mean AUC, certain labels):")
    for f, p in zip(args.pred_files, pred_matrices):
        m = compute_chexpert_auc(y_true, p, valid_mask=~unc)
        print(f"    * {Path(f).name:<52s} {m['competition_mean_auc']:.4f}")

    if args.method == "mean":
        ensemble_preds = np.mean(pred_matrices, axis=0)
    elif args.method == "weighted":
        w = np.array(args.weights if args.weights is not None else [1.0] * len(pred_matrices), dtype=np.float32)
        if len(w) != len(pred_matrices):
            raise SystemExit("--weights needs one value per prediction file.")
        w = w / w.sum()
        print(f"  * Normalized Weights: {w.tolist()}")
        ensemble_preds = sum(wi * p for wi, p in zip(w, pred_matrices))
    else:
        ensemble_preds = rank_average(pred_matrices)

    metrics = compute_chexpert_auc(y_true, ensemble_preds, valid_mask=~unc)
    print_auc_summary(metrics, f"Ensemble Results ({args.split.upper()} split - {args.method.upper()})")

    out_path = out_dir / f"ensemble_{args.method}_{args.split}_predictions.csv"
    predictions_frame(y_true, ensemble_preds, unc).to_csv(out_path, index=False)
    print(f"\nEnsemble predictions saved to: {out_path}")


if __name__ == "__main__":
    run_ensemble()

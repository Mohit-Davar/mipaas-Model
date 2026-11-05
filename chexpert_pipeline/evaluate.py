"""
evaluate.py
=============================================================================
Standardized Evaluation & Calibration Engine for CheXpert Plus Models.
=============================================================================
Features:
- Checkpoint loading (.pth) and model reconstruction from the checkpoint's own settings
  (model name, image size, dropout, chunks).
- Inference on the full Tune, Test, or official CheXpert Valid partition.
- Optional horizontal-flip test-time augmentation (--tta).
- Per-pathology ROC-AUC on certain labels and competition benchmark metrics.
- Optimal operating thresholds (Youden's J) computed on certain labels.
- Prediction CSV (pred_/true_/unc_ columns, ready for ensemble.py) and JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import (
    ALL_LABELS,
    COMPETITION_PATHOLOGIES,
    CONFIG,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    get_device,
)
from dataset import create_eval_loader
from models import build_model, model_normalization
from metrics import compute_chexpert_auc, find_optimal_thresholds, print_auc_summary, predictions_frame


def parse_args():
    parser = argparse.ArgumentParser(description="CheXpert Plus Standardized Evaluation Engine")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="valid", choices=["tune", "test", "valid"],
                        help="'valid' = official CheXpert validation set (radiologist labels)")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_limit", type=int, default=None, help="Sample limit for fast inspection")
    parser.add_argument("--tta", action="store_true", help="Average predictions with the horizontally flipped image")
    parser.add_argument("--cache_dir", type=str, default="auto",
                        help="'auto' = smallest image cache that matches the checkpoint's resolution")
    parser.add_argument("--out_dir", type=str, default=str(OUTPUT_DIR), help="Output directory for predictions")
    return parser.parse_args()


@torch.no_grad()
def run_evaluation():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Check inside CHECKPOINT_DIR
        candidate = CHECKPOINT_DIR / ckpt_path.name
        if candidate.exists():
            ckpt_path = candidate
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {}) or {}
    model_name = checkpoint.get("model_name", saved_args.get("model_name", "tri_nn_transformer"))
    run_name = checkpoint.get("run_name", saved_args.get("run_name") or model_name)
    image_size = int(saved_args.get("image_size", CONFIG["image_size"][0]))

    model = build_model(
        model_name=model_name,
        num_classes=len(ALL_LABELS),
        pretrained=False,
        dropout=saved_args.get("dropout", CONFIG["dropout"]),
        head_hidden=saved_args.get("head_hidden", CONFIG["head_hidden"]),
        drop_path_rate=saved_args.get("drop_path_rate", CONFIG["drop_path_rate"]),
        image_size=image_size,
        freeze_blocks=saved_args.get("freeze_blocks", 0),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    loader = create_eval_loader(
        args.split,
        chunks=saved_args.get("chunks"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(image_size, image_size),
        sample_limit=args.sample_limit,
        cache_dir=args.cache_dir or None,
        norm=model_normalization(model_name),
    )
    if loader is None:
        raise SystemExit(f"Split '{args.split}' has no images in the manifest. "
                         f"(The 'valid' split exists only after tools/build_manifest_chexpert.py.)")

    print(f"Running inference on {args.split.upper()} split ({len(loader.dataset):,} images, {image_size}px"
          f"{', hflip TTA' if args.tta else ''})...")
    all_preds, all_targets, all_unc = [], [], []

    for images, targets, unc in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            probs = torch.sigmoid(model(images).float())
            if args.tta:
                probs = 0.5 * (probs + torch.sigmoid(model(torch.flip(images, dims=[3])).float()))

        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_unc.append(unc.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    unc_arr = np.concatenate(all_unc, axis=0)

    # Compute official metrics (certain labels only)
    metrics = compute_chexpert_auc(y_true, y_pred, valid_mask=(~unc_arr))
    print_auc_summary(metrics, f"Evaluation Results ({args.split.upper()} Split)")

    # Optimize decision thresholds on certain labels
    thresholds = find_optimal_thresholds(y_true, y_pred, valid_mask=(~unc_arr))
    print("\nOptimal Operating Thresholds (Youden's J Statistic):")
    for lbl in COMPETITION_PATHOLOGIES:
        print(f"  * {lbl:<26s}: {thresholds[lbl]:.4f}")

    # Export predictions to CSV
    tag = f"{run_name}_{args.split}{'_tta' if args.tta else ''}"
    pred_df = predictions_frame(y_true, y_pred, unc_arr)
    out_csv = out_dir / f"{tag}_predictions.csv"
    pred_df.to_csv(out_csv, index=False)
    print(f"\nPredictions exported to: {out_csv}")

    # Export calibration report
    report = {
        "model_name": model_name,
        "run_name": run_name,
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "tta": bool(args.tta),
        "image_size": image_size,
        "samples_evaluated": len(pred_df),
        "competition_mean_auc": metrics["competition_mean_auc"],
        "mean_auc_all": metrics["mean_auc_all"],
        "pred_std": metrics["pred_std"],
        "thresholds": thresholds,
        "per_class_auc": metrics["per_class_auc"],
        "per_class_support": {k: list(v) for k, v in metrics["per_class_support"].items()},
    }
    out_json = out_dir / f"{tag}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics report exported to: {out_json}")


if __name__ == "__main__":
    run_evaluation()

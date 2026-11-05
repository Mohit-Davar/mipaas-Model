"""
verify_pipeline.py
=============================================================================
Comprehensive Automated Pipeline Verification Suite for CheXpert Plus.
=============================================================================
Runs an end-to-end sanity verification on the standardized codebase:
  1. Configuration & Path Resolution
  2. Dataset Loading & DataLoader Batch Generation
  3. Model Instantiation & Parameter Count
  4. Forward Pass & Output Shape Integrity
  5. Output Bias Initialization to Class Log-Odds
  6. Backward Pass & Full Gradient Flow Across All 3 Backbones
  7. Prediction Dispersion & Collapse Prevention
  8. Metric Computation & ROC-AUC Summary Formatting
"""

from __future__ import annotations

import os
import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ALL_LABELS,
    COMPETITION_PATHOLOGIES,
    CONFIG,
    BASE_DIR,
    DATASET_DIR,
    MANIFEST_PARQUET,
    get_device,
    set_seed,
)
from dataset import create_dataloaders
from models import build_model, TriBackboneFusion
from losses import build_loss
from metrics import compute_chexpert_auc, print_auc_summary


def run_full_verification() -> bool:
    print("=" * 72)
    print(" CheXpert Plus: Automated Pipeline Verification Suite")
    print("=" * 72)
    t0 = time.time()
    set_seed(42)
    device = get_device()
    print(f"Executing on device: {device}\n")

    # -----------------------------------------------------------------------
    # Test 1: Configuration & Path Resolution
    # -----------------------------------------------------------------------
    print("[1/8] Verifying Configuration & Path Resolution...")
    print(f"  * BASE_DIR    : {BASE_DIR}")
    print(f"  * DATASET_DIR : {DATASET_DIR}")
    print(f"  * MANIFEST    : {MANIFEST_PARQUET}")

    if not MANIFEST_PARQUET.exists():
        print(f"  [FAIL] Manifest parquet file not found at {MANIFEST_PARQUET}")
        return False
    if not DATASET_DIR.exists():
        print(f"  [FAIL] Dataset image directory not found at {DATASET_DIR}")
        return False
    print("  >>> PASS: Paths resolved successfully.")

    # -----------------------------------------------------------------------
    # Test 2: Dataset Loading & Batch Generation
    # -----------------------------------------------------------------------
    print("\n[2/8] Verifying DataLoader & Batch Generation (Sample Subset)...")
    train_loader, tune_loader, _ = create_dataloaders(
        manifest_path=MANIFEST_PARQUET,
        batch_size=4,
        num_workers=0,
        sample_limit=32,
        tune_sample_limit=16,
        seed=42,
    )

    batch_images, batch_targets, batch_unc = next(iter(train_loader))
    print(f"  * Image batch shape  : {batch_images.shape} (dtype: {batch_images.dtype})")
    print(f"  * Target batch shape : {batch_targets.shape} (dtype: {batch_targets.dtype})")
    print(f"  * Mask batch shape   : {batch_unc.shape} (dtype: {batch_unc.dtype})")

    assert batch_images.shape == (4, 3, 384, 384), f"Unexpected image shape: {batch_images.shape}"
    assert batch_targets.shape == (4, len(ALL_LABELS)), f"Unexpected target shape: {batch_targets.shape}"
    assert torch.isfinite(batch_images).all(), "Image batch contains NaN/Inf values!"
    print("  >>> PASS: DataLoader batch successfully instantiated.")

    # -----------------------------------------------------------------------
    # Test 3: Model Architecture & Parameter Audit
    # -----------------------------------------------------------------------
    print("\n[3/8] Verifying TriBackboneFusion Architecture & Parameters...")
    model = build_model(
        model_name="tri_nn_transformer",
        num_classes=len(ALL_LABELS),
        pretrained=True,
        dropout=0.3,
        head_hidden=768,
        drop_path_rate=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  * Total parameters     : {total_params / 1e6:.1f}M ({total_params:,d})")
    print(f"  * Trainable parameters : {trainable_params / 1e6:.1f}M")
    
    assert total_params > 100_000_000, f"Parameter count lower than expected: {total_params}"
    assert isinstance(model, TriBackboneFusion), f"Model is not an instance of TriBackboneFusion: {type(model)}"
    print("  >>> PASS: TriBackboneFusion instantiated correctly.")

    # -----------------------------------------------------------------------
    # Test 4: Forward Pass & Output Shape Integrity
    # -----------------------------------------------------------------------
    print("\n[4/8] Testing Forward Pass & Output Dimension...")
    dummy_input = batch_images.to(device)
    with torch.no_grad():
        dummy_logits = model(dummy_input)

    print(f"  * Output logits shape: {dummy_logits.shape} (expected: [4, {len(ALL_LABELS)}])")
    assert dummy_logits.shape == (4, len(ALL_LABELS)), f"Logits shape mismatch: {dummy_logits.shape}"
    assert torch.isfinite(dummy_logits).all(), "Forward pass produced NaN/Inf logits!"
    print("  >>> PASS: Forward pass verified.")

    # -----------------------------------------------------------------------
    # Test 5: Output Bias Initialization to Class Log-Odds
    # -----------------------------------------------------------------------
    print("\n[5/8] Testing Output Bias Initialization to Empirical Priors...")
    train_ds = train_loader.dataset
    prior = train_ds.labels.float().mean(dim=0).clamp(min=1e-3, max=1.0 - 1e-3)
    model.set_output_bias(prior.to(device))

    # Retrieve output linear layer directly
    last_linear = None
    for m in model.head.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    assert last_linear is not None, "Output Linear layer not found in model.head!"

    actual_bias = last_linear.bias.cpu()
    expected_log_odds = torch.log(prior / (1.0 - prior))
    bias_diff = (actual_bias - expected_log_odds).abs().max().item()
    assert bias_diff < 1e-4, f"Output bias mismatch: max error = {bias_diff}"

    bias_probs = torch.sigmoid(actual_bias)
    print("  * Bias prob vs Empirical train prevalence (first 5 classes):")
    for idx, name in enumerate(ALL_LABELS[:5]):
        expected = prior[idx].item()
        actual = bias_probs[idx].item()
        print(f"    - {name:<26s} | Expected: {expected:.3f} | Initial Bias: {actual_bias[idx].item():+.3f} -> Prob: {actual:.3f}")
        assert abs(expected - actual) < 0.01, f"Bias mismatch for {name}: {expected} vs {actual}"
    print("  >>> PASS: Output bias successfully initializes model to training priors.")

    # -----------------------------------------------------------------------
    # Test 6: Backward Pass & Full Gradient Highway
    # -----------------------------------------------------------------------
    print("\n[6/8] Verifying Backward Pass & Gradient Flow Across All Backbones...")
    model.train()
    criterion = build_loss("bce").to(device)
    dummy_targets = batch_targets.to(device)

    logits = model(dummy_input)
    loss = criterion(logits, dummy_targets)
    loss.backward()

    # Audit critical checkpoint layers
    key_layers = [
        ("DenseNet Conv0 (Early Layer)", model.densenet.features.conv0.weight),
        ("DenseNet Block4 (Deep Layer)", list(model.densenet.features.denseblock4.parameters())[0]),
        ("ConvNeXt Stage 3 (Deep Layer)", list(model.convnext.stages[3].parameters())[0]),
        ("EfficientNet Block 6 (Deep)", list(model.efficientnet.blocks[6].parameters())[0]),
        ("Fusion BatchNorm1d", model.fusion_bn.weight),
        ("Output Linear Projection Layer", list(model.head.children())[-1].weight),
    ]

    for label, param in key_layers:
        grad = param.grad
        assert grad is not None, f"[FAIL] No gradient flowed to {label}!"
        grad_norm = grad.norm().item()
        assert grad_norm > 1e-7, f"[FAIL] Gradient vanished for {label}: norm={grad_norm}"
        print(f"  * {label:<32s}: grad_norm = {grad_norm:.6f} [ACTIVE]")
    print("  >>> PASS: Gradients flow unimpeded through all backbones and head.")

    # -----------------------------------------------------------------------
    # Test 7: Prediction Dispersion & Collapse Prevention
    # -----------------------------------------------------------------------
    print("\n[7/8] Testing Prediction Variance Across Different Inputs...")
    model.eval()
    with torch.no_grad():
        # Evaluate on the batch of 4 distinct images
        pred_probs = torch.sigmoid(model(dummy_input)).cpu().numpy()
        pred_std_across_images = float(np.mean(np.std(pred_probs, axis=0)))

    print(f"  * Prediction standard deviation across images: {pred_std_across_images:.4f}")
    assert pred_std_across_images > 0.01, f"[FAIL] Static prediction collapse detected! std={pred_std_across_images}"
    print("  >>> PASS: Model produces distinct, image-dependent predictions.")

    # -----------------------------------------------------------------------
    # Test 8: Evaluation Metrics & ROC-AUC Engine
    # -----------------------------------------------------------------------
    print("\n[8/8] Testing Metric Calculation Engine...")
    # Create synthetic test ground-truth with mixed classes
    k = len(ALL_LABELS)
    y_true_mock = np.array([
        [1 if (i + j) % 2 == 0 else 0 for j in range(k)] for i in range(4)
    ], dtype=np.float32)
    y_pred_mock = np.array([
        [0.85 if y_true_mock[i, j] == 1 else 0.15 for j in range(k)] for i in range(4)
    ], dtype=np.float32)

    test_metrics = compute_chexpert_auc(y_true_mock, y_pred_mock)
    print(f"  * Mock Competition Mean AUC : {test_metrics['competition_mean_auc']:.4f}")
    print(f"  * Mock Mean AUC (All {k})    : {test_metrics['mean_auc_all']:.4f}")
    assert test_metrics["competition_mean_auc"] > 0.80, "ROC-AUC calculation gave unexpected result"
    print("  >>> PASS: Metric engine functions accurately.")

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f" ALL 8 PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY! ({elapsed:.2f}s)")
    print(" The standardized codebase is mathematically verified and ready to train.")
    print("=" * 72)
    return True


if __name__ == "__main__":
    success = run_full_verification()
    sys.exit(0 if success else 1)

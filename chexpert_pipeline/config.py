"""
config.py
=============================================================================
Central Configuration Module for Standardized CheXpert Plus Classification.
=============================================================================
This module defines all global dataset constants, paths, default training
hyperparameters, label registries, and hardware initialization routines.
"""

from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Windows Runtime Environment Optimization
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Configure UTF-8 encoding on standard output for crash-free console logging
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Filesystem Paths Resolution
# ---------------------------------------------------------------------------
# SCRIPT_DIR is d:\Yatharth\Mipaas\chexpert_pipeline
SCRIPT_DIR = Path(__file__).resolve().parent
# BASE_DIR is d:\Yatharth\Mipaas (parent directory housing dataset & manifest)
DEFAULT_BASE_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "split_manifest.parquet").exists() else SCRIPT_DIR

BASE_DIR = Path(os.environ.get("CHEXPERT_BASE_DIR", DEFAULT_BASE_DIR)).resolve()
DATASET_DIR = Path(os.environ.get("CHEXPERT_DATA_DIR", BASE_DIR / "ChestXray Dataset")).resolve()

MANIFEST_PARQUET = BASE_DIR / "split_manifest.parquet"
MANIFEST_CSV = BASE_DIR / "split_manifest.csv"

# Output and checkpoint directories (can be within BASE_DIR or custom)
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Original CheXpert v1.0 label CSVs (train.csv / valid.csv) go here.
# They replace the CheXbert JSONs (report_fixed.json etc.), whose labels do not
# line up with their own path_to_image column. See tools/build_manifest_chexpert.py.
LABELS_DIR = Path(os.environ.get("CHEXPERT_LABELS_DIR", BASE_DIR / "labels")).resolve()

# Pre-resized image caches built by tools/build_image_cache.py, one folder per size:
#   C:\chexpert_cache\jpg512, C:\chexpert_cache\jpg1024, ...
# The originals are 2000-3400 px PNGs (3-9 MB each) on a SATA hard drive (D:), which
# caps loading at ~20-30 img/s. A JPEG cache on the NVMe drive (C:) loads at several
# hundred img/s, so the GPU becomes the bottleneck. Training picks the smallest cache
# whose long side >= the training resolution ("auto"); CHEXPERT_CACHE_DIR forces one folder.
CACHE_ROOT = Path(os.environ.get("CHEXPERT_CACHE_ROOT", r"C:\chexpert_cache"))
CACHE_DIR = Path(os.environ.get("CHEXPERT_CACHE_DIR", CACHE_ROOT / "jpg512"))


def pick_cache_dir(image_size: int):
    """Smallest cache folder (jpg<side>) under CACHE_ROOT with side >= image_size, else None."""
    if os.environ.get("CHEXPERT_CACHE_DIR"):
        return Path(os.environ["CHEXPERT_CACHE_DIR"])
    best = None
    if CACHE_ROOT.is_dir():
        for d in CACHE_ROOT.iterdir():
            name = d.name.lower()
            if d.is_dir() and name.startswith("jpg") and name[3:].isdigit():
                side = int(name[3:])
                if side >= 0.98 * image_size and (best is None or side < best[0]):  # <=2% upscale is harmless
                    best = (side, d)
    return best[1] if best else None


def cache_path_for(file_path, cache_dir=None) -> Path:
    """Maps an original PNG path to its cached JPEG (mirrors the chunk-relative path)."""
    s = str(file_path).replace("/", "\\")
    i = s.lower().find("png_chexpert_plus_chunk_")
    rel = s[i:] if i >= 0 else Path(s).name
    return (Path(cache_dir or CACHE_DIR) / rel).with_suffix(".jpg")

# ---------------------------------------------------------------------------
# Label Registries (Target Pathologies + Competition Benchmark Classes)
# ---------------------------------------------------------------------------
# Full 14 raw CheXpert annotations reference
FULL_14_LABELS: List[str] = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

# Active Target Pathologies (7 classes: 5 competition targets + 2 high-prevalence markers)
# Removed 7 severe-imbalance classes: Pneumonia, Enlarged Cardiomediastinum,
# Pneumothorax, Lung Lesion, No Finding, Fracture, Pleural Other.
ALL_LABELS: List[str] = [
    "Cardiomegaly",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
    "Support Devices",
]

COMPETITION_PATHOLOGIES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

COMPETITION_INDICES: List[int] = [ALL_LABELS.index(p) for p in COMPETITION_PATHOLOGIES]

# Per-class handling of uncertain (-1) labels for uncertainty_policy='mixed'.
# CheXpert paper (Irvin et al., 2019): U-Ones works best for Atelectasis and Edema,
# U-Zeros-style mappings for Cardiomegaly and Consolidation.
UNCERTAINTY_PER_CLASS: Dict[str, str] = {
    "Cardiomegaly": "zeros",
    "Lung Opacity": "ones",
    "Edema": "ones",
    "Consolidation": "zeros",
    "Atelectasis": "ones",
    "Pleural Effusion": "ones",
    "Support Devices": "ones",
}

# Label-smoothing ranges for uncertain labels (Pham et al., 2021: U-Ones+LSR / U-Zeros+LSR).
# Uncertain targets become a fixed random value in this range instead of a hard 0/1.
LSR_RANGES: Dict[str, tuple] = {"ones": (0.55, 0.85), "zeros": (0.0, 0.30)}

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Default Pipeline Hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Data & Resolution (512 px; RAD-DINO uses its native 518 px, see run_training.bat)
    "image_size": (512, 512),
    "num_classes": len(ALL_LABELS),
    "frontal_only": True,
    "uncertainty_policy": "mixed",   # 'mixed' (per-class, see UNCERTAINTY_PER_CLASS) | 'u_ones' | 'u_zeros' | 'u_ignore'
    "label_smoothing": True,         # Soft targets for uncertain labels (LSR_RANGES), training split only
    "chunks": [0, 1, 2, 3, 4],       # Manual import of chunks 0, 1, 2, 3, 4 (chunk_4_3)

    # Batch & Parallelism (measured on RTX 4070 SUPER 12 GB, AMP, NCHW, GPU-only; epoch = 172k images):
    #   512 px  densenet121          bs32: 10.5 GB, 114 img/s (~25 min/epoch)  -> bs16 x2 used for headroom
    #   512 px  tf_efficientnetv2_s  bs16:  6.3 GB, 120 img/s (~24 min/epoch)  (bs32 OOM)
    #   512 px  convnext_small       bs16:  8.3 GB,  75 img/s (~38 min/epoch)  (bs32 OOM)
    #   518 px  rad_dino (ViT-B/14)  bs8 full fine-tune: 5.1 GB, 35 img/s (~82 min/epoch)
    #                                frozen encoder (head only): 1.6 GB, ~136 img/s (~21 min/epoch)
    #   384 px  tri_nn_transformer   bs8: 8.0 GB, ~42 img/s (~61 min/epoch); bs12 does NOT fit
    "batch_size": 16,
    "accum_steps": 2,                # effective batch 32
    "num_workers": 8,                # Needs the image cache; with full-res PNGs on the HDD the disk is the limit
    "pin_memory": True,
    "max_vram_fraction": 0.92,       # Raise OOM instead of silently spilling into system RAM (Windows sysmem fallback)

    # Optimization & Learning Rates
    "backbone_lr": 1e-4,
    "head_lr": 1e-3,
    "weight_decay": 1e-2,
    "warmup_epochs": 0.2,
    "min_lr_ratio": 0.01,            # Cosine decay down to 1% of peak LR
    "gradient_clip": 1.0,            # Max L2 gradient norm
    "amp_enabled": True,             # Automatic Mixed Precision (torch.cuda.amp)

    # Loss & Objective
    "loss": "bce",                   # Masked BCE (accepts soft targets)

    # Model Architecture: one strong single backbone per run, then ensemble.py across runs
    # (densenet121 + tf_efficientnetv2_s + convnext_small). 'tri_nn_transformer' is still available.
    "model_name": "densenet121",
    "dropout": 0.2,                  # Classifier dropout (tri model: MLP projection dropout)
    "head_hidden": 768,              # Tri model only: hidden dimension of 2-layer MLP projection
    "drop_path_rate": 0.1,           # Stochastic depth for ConvNeXt, EfficientNet & RAD-DINO
    "grad_checkpointing": False,     # Only needed if VRAM < 10GB
    "freeze_blocks": 0,              # RAD-DINO: 0 = full fine-tune, 12 = frozen encoder (head only)
    "layer_decay": 0.75,             # RAD-DINO: layer-wise LR decay (block i gets lr * 0.75^(13-i))

    # Schedule & Stopping
    "epochs": 3,                     # CheXpert models usually peak within 2-3 epochs
    "evals_per_epoch": 3,            # Validate (and keep the best checkpoint) 3x per epoch
    "patience": 2,                   # Early stopping patience, in epochs
    "tune_sample_limit": 5000,       # Intermediate validation subset (full tune/test/valid are used at the end)
    
    # Augmentation
    "hflip": True,                   # Random horizontal flip
    "rotation": 7.0,                 # Random rotation degrees
    "jitter": 0.1,                   # Color jitter intensity
    
    # Reproducibility
    "seed": 42,
}

CONFIG = DEFAULT_CONFIG.copy()


def ensure_dirs() -> None:
    """Creates required output and checkpoint directories if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    """Returns the primary torch.device (CUDA if available, else CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_seed(seed: int = CONFIG["seed"]) -> None:
    """Sets deterministic random seeds across python, numpy, and torch."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


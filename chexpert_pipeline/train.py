"""
train.py
=============================================================================
Standardized Training & Evaluation Engine for CheXpert Plus.
=============================================================================
Features:
- Single timm backbones (densenet121, tf_efficientnetv2_s, convnext_small, ...) or
  the TriBackboneFusion model ('tri_nn_transformer').
- Automatic Mixed Precision (AMP) with PyTorch GradScaler.
- Discriminative learning rates with bias & normalization weight decay exclusion.
- Empirical class log-odds output bias initialization.
- Validation several times per epoch, keeping the best checkpoint by competition AUC.
- VRAM cap: raises OOM instead of silently spilling into system RAM (Windows driver fallback).
- Final evaluation on the held-out test split and the official CheXpert validation set,
  with prediction CSVs ready for ensemble.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from typing import Dict, Any, Tuple, Optional, List, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import (
    ALL_LABELS,
    CHECKPOINT_DIR,
    CONFIG,
    MANIFEST_PARQUET,
    OUTPUT_DIR,
    ensure_dirs,
    get_device,
    set_seed,
)
from dataset import create_dataloaders, create_eval_loader
from losses import build_loss, compute_pos_weight
from metrics import compute_chexpert_auc, print_auc_summary, predictions_frame
from models import build_model, model_normalization


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CheXpert Plus Standardized Training Engine")

    # Model
    parser.add_argument("--model_name", type=str, default=CONFIG["model_name"],
                        help="timm backbone (densenet121, tf_efficientnetv2_s, convnext_small, ...) or tri_nn_transformer")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Prefix for checkpoints/outputs (default: model_name). Use it to train several seeds.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=CONFIG["dropout"])
    parser.add_argument("--head_hidden", type=int, default=CONFIG["head_hidden"])
    parser.add_argument("--drop_path_rate", type=float, default=CONFIG["drop_path_rate"])
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--freeze_blocks", type=int, default=CONFIG["freeze_blocks"],
                        help="rad_dino: number of transformer blocks to freeze (0 = full fine-tune, 12 = head only)")
    parser.add_argument("--layer_decay", type=float, default=CONFIG["layer_decay"],
                        help="rad_dino: layer-wise LR decay; block i gets backbone_lr * layer_decay^(13 - i)")

    # Training & Optimization
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--accum_steps", type=int, default=CONFIG["accum_steps"])
    parser.add_argument("--image_size", type=int, default=CONFIG["image_size"][0])
    parser.add_argument("--backbone_lr", type=float, default=CONFIG["backbone_lr"])
    parser.add_argument("--head_lr", type=float, default=CONFIG["head_lr"])
    parser.add_argument("--weight_decay", type=float, default=CONFIG["weight_decay"])
    parser.add_argument("--warmup_epochs", type=float, default=CONFIG["warmup_epochs"])
    parser.add_argument("--min_lr_ratio", type=float, default=CONFIG["min_lr_ratio"])
    parser.add_argument("--grad_clip", type=float, default=CONFIG["gradient_clip"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=CONFIG["amp_enabled"])
    parser.add_argument("--max_vram_fraction", type=float, default=CONFIG["max_vram_fraction"],
                        help="Cap on GPU memory (0-1). Prevents the Windows driver from spilling into system RAM.")

    # Objective & Uncertainty
    parser.add_argument("--loss", type=str, default=CONFIG["loss"], choices=["bce", "asl"])
    parser.add_argument("--uncertainty_policy", type=str, default=CONFIG["uncertainty_policy"],
                        choices=["mixed", "u_ones", "u_zeros", "u_ignore"])
    parser.add_argument("--label_smoothing", action=argparse.BooleanOptionalAction, default=CONFIG["label_smoothing"],
                        help="Soft targets for uncertain labels (training split only)")
    parser.add_argument("--use_pos_weight", action="store_true")

    # Data & Validation
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PARQUET))
    parser.add_argument("--chunks", nargs="+", default=["0", "1", "2", "3", "4"],
                        help="Specific dataset chunks to load manually (e.g. --chunks 0 1 2 3 4)")
    parser.add_argument("--cache_dir", type=str, default="auto",
                        help="Image cache folder; 'auto' = smallest C:\\chexpert_cache\\jpg<side> with side >= "
                             "--image_size. Empty string disables it.")
    parser.add_argument("--tune_sample_limit", type=int, default=CONFIG.get("tune_sample_limit", 5000))
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    parser.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=CONFIG["hflip"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--evals_per_epoch", type=int, default=CONFIG.get("evals_per_epoch", 3))
    parser.add_argument("--patience", type=int, default=CONFIG.get("patience", 2),
                        help="Early stopping patience in epochs (0 disables)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_test", action="store_true")

    return parser.parse_args(argv)


def _no_decay(name: str, p: torch.nn.Parameter) -> bool:
    """Returns True if parameter should be excluded from weight decay (biases, normalizations)."""
    return p.ndim <= 1 or "bias" in name or "norm" in name or "bn" in name


def build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    layer_decay: float = 0.75,
) -> AdamW:
    """
    Constructs an AdamW optimizer with parameter groups segregating backbones and head,
    and excluding 1D parameters from weight decay. Models with their own param_groups()
    (RAD-DINO: layer-wise LR decay) define the groups themselves.
    """
    if hasattr(model, "param_groups"):
        groups = [g for g in model.param_groups(backbone_lr, head_lr, weight_decay, layer_decay) if g["params"]]
        lrs = [g["lr"] for g in groups if not g["name"].startswith("head")]
        n_train = sum(p.numel() for g in groups for p in g["params"])
        print(f"  Optimizer: {len(groups)} groups, {n_train:,d} trainable params | layer-wise LR "
              f"{min(lrs):.1e} (lowest layer) -> {max(lrs):.1e} (top) | head lr {head_lr:.1e}" if lrs else
              f"  Optimizer: head only ({n_train:,d} params) | head lr {head_lr:.1e}")
        return AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    if hasattr(model, "split_params"):
        body, head = model.split_params()
    else:
        body = list(model.named_parameters())
        head = []

    groups = []
    for params, lr, tag in ((body, backbone_lr, "backbone"), (head, head_lr, "head")):
        if not params:
            continue
        decay = [p for n, p in params if not _no_decay(n, p)]
        plain = [p for n, p in params if _no_decay(n, p)]
        if decay:
            groups.append({
                "params": decay,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": f"{tag}_decay",
            })
        if plain:
            groups.append({
                "params": plain,
                "lr": lr,
                "weight_decay": 0.0,
                "name": f"{tag}_no_decay",
            })

    print("  Optimizer parameter groups:")
    for g in groups:
        count = sum(p.numel() for p in g["params"])
        print(f"    * {g['name']:<18s} tensors={len(g['params']):>4d} | params={count:>12,d} | lr={g['lr']:.1e} | wd={g['weight_decay']}")

    return AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> LambdaLR:
    """Constructs a cosine annealing schedule with linear warmup."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    ignore_uncertain: bool,
    accum_steps: int,
    grad_clip: float,
    epoch: int = 1,
    eval_every: int = 0,
    eval_fn: Optional[Callable[[float, float], bool]] = None,
) -> Tuple[float, bool]:
    """
    Executes a single training epoch. Every `eval_every` optimizer steps it calls
    eval_fn(epoch_progress, train_loss); if that returns True, training stops early.
    Returns (mean train loss, stopped_early).
    """
    model.train()
    total_steps = len(loader)
    log_interval = max(1, min(200, total_steps // 20))
    running_loss = torch.zeros((), device=device)
    seen_samples, skipped_steps, opt_steps = 0, 0, 0
    t0 = time.time()
    t_steady, seen_at_steady = None, 0  # speed/ETA exclude the one-time startup (worker spawn, cuDNN autotune)

    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets, unc) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = (~unc).to(device, non_blocking=True).float() if ignore_uncertain else None

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets, mask)

        scaler.scale(loss / accum_steps).backward()

        bs = images.size(0)
        running_loss += loss.detach() * bs
        seen_samples += bs

        last_step = (step + 1) == total_steps
        stepped = (step + 1) % accum_steps == 0 or last_step
        if stepped:
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) if grad_clip > 0 else None

            if use_amp or gnorm is None or torch.isfinite(gnorm):
                scaler.step(optimizer)
            else:
                skipped_steps += 1

            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Suppress false-positive warning from PyTorch LRScheduler when used with GradScaler
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scheduler.step()
            opt_steps += 1

        if step == 0:
            mem = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            print(f"  [Initialization complete: Step 1 finished in {time.time() - t0:.1f}s | memory={mem:.0f}MB]", flush=True)
            t_steady, seen_at_steady = time.time(), seen_samples

        if (step + 1) % log_interval == 0 or last_step:
            now = time.time()
            if step > 0:
                steady = max(1e-3, now - t_steady)
                rate = (seen_samples - seen_at_steady) / steady
                eta_min = (total_steps - step - 1) * steady / step / 60
            else:
                rate, eta_min = seen_samples / max(1e-3, now - t0), float("nan")
            lr_str = "/".join(f"{g['lr']:.1e}" for g in optimizer.param_groups[::max(1, len(optimizer.param_groups) // 2)])
            avg_loss = running_loss.item() / max(1, seen_samples)
            print(f"  Step [{step + 1:>5d}/{total_steps:>5d}] loss {avg_loss:.4f} | {rate:>5.1f} img/s | lr {lr_str} | ETA {eta_min:.0f} min"
                  + (f" | skipped {skipped_steps}" if skipped_steps else ""), flush=True)

        if (eval_fn is not None and eval_every > 0 and stepped and not last_step
                and opt_steps % eval_every == 0):
            stop = eval_fn(epoch - 1 + (step + 1) / total_steps, running_loss.item() / max(1, seen_samples))
            model.train()
            if stop:
                return running_loss.item() / max(1, seen_samples), True

    return running_loss.item() / max(1, seen_samples), False


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference across a validation or test DataLoader."""
    model.eval()
    loss_sum, n_samples = 0.0, 0
    all_targets, all_preds, all_unc = [], [], []

    for images, targets, unc in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # During evaluation, always apply uncertainty mask regardless of training policy.
        # This ensures AUC is computed only on certain annotations (ground-truth quality).
        mask = (~unc).to(device, non_blocking=True).float()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets, mask)

        loss_sum += loss.item() * images.size(0)
        n_samples += images.size(0)

        all_targets.append(targets.cpu().numpy())
        all_preds.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_unc.append(unc.numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    unc_arr = np.concatenate(all_unc, axis=0)

    metrics = compute_chexpert_auc(y_true, y_pred, valid_mask=(~unc_arr))
    metrics["competition_mean_auc_mapped"] = compute_chexpert_auc(y_true, y_pred)["competition_mean_auc"]
    metrics["val_loss"] = loss_sum / max(1, n_samples)

    return metrics, y_true, y_pred, unc_arr


def _plain(obj: Any) -> Any:
    """Serializes numpy and scalar values into standard JSON-compatible Python types."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def run_training(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    run_name = args.run_name or args.model_name
    ensure_dirs()
    set_seed(args.seed)
    device = get_device()
    use_amp = bool(args.amp) and (device.type == "cuda")
    cache_dir = args.cache_dir or None

    ignore_uncertain = (args.uncertainty_policy == "u_ignore")

    if device.type == "cuda" and 0 < args.max_vram_fraction < 1:
        # Windows drivers page GPU memory into system RAM when VRAM runs out (10x+ slower, no error).
        torch.cuda.set_per_process_memory_fraction(args.max_vram_fraction, device)

    print("=" * 78)
    print(" CheXpert Plus Standardized Training Pipeline")
    print(f" Run: {run_name} | Model: {args.model_name} | Pretrained: {args.pretrained}")
    print(f" Device: {device} | AMP: {use_amp} | Image Size: {args.image_size}x{args.image_size} | VRAM cap: {args.max_vram_fraction:.0%}")
    print(f" Batch: {args.batch_size}x{args.accum_steps} (Effective: {args.batch_size * args.accum_steps}) | Epochs: {args.epochs} | Evals/epoch: {args.evals_per_epoch}")
    print(f" Policy: {args.uncertainty_policy}{' +LSR' if args.label_smoothing else ''} | Loss: {args.loss}")
    print(f" Learning Rates: Backbone={args.backbone_lr:.1e} | Head={args.head_lr:.1e} | WD={args.weight_decay:.1e}")
    print(f" Architecture: Dropout={args.dropout} | DropPath={args.drop_path_rate}"
          + (f" | Head Hidden={args.head_hidden}" if "tri" in args.model_name.lower() else "")
          + (f" | Frozen blocks={args.freeze_blocks} | Layer decay={args.layer_decay}"
             if "dino" in args.model_name.lower() else ""))
    print("=" * 78)

    # 1. Dataloaders (input normalization must match the model's pretraining)
    norm = model_normalization(args.model_name)
    tune_limit = args.tune_sample_limit if args.tune_sample_limit > 0 else None
    train_loader, tune_loader, test_loader = create_dataloaders(
        manifest_path=args.manifest,
        chunks=args.chunks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_size, args.image_size),
        uncertainty_policy=args.uncertainty_policy,
        sample_limit=args.sample_limit,
        tune_sample_limit=tune_limit,
        seed=args.seed,
        hflip=args.hflip,
        label_smoothing=args.label_smoothing,
        cache_dir=cache_dir,
        norm=norm,
    )
    train_ds = train_loader.dataset

    # 2. Model Creation
    model = build_model(
        model_name=args.model_name,
        num_classes=CONFIG["num_classes"],
        pretrained=args.pretrained,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
        drop_path_rate=args.drop_path_rate,
        grad_checkpointing=args.grad_checkpointing,
        image_size=args.image_size,
        freeze_blocks=args.freeze_blocks,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters: {total_params / 1e6:.1f}M total | {trainable_params / 1e6:.1f}M trainable")

    # 3. Output Bias Initialization
    # Initialize output layer bias to empirical training class log-odds
    if hasattr(model, "set_output_bias"):
        prior = train_ds.labels.float().mean(dim=0).clamp(min=1e-3, max=1.0 - 1e-3)
        model.set_output_bias(prior.to(device))
        print(f"  Output bias initialized to training class priors ({args.uncertainty_policy}):")
        for name, p_val in zip(ALL_LABELS, prior.tolist()):
            print(f"    * {name:<28s}: {p_val * 100:>5.1f}%")

    # 4. Loss & Optimizer
    pos_weight = None
    if args.use_pos_weight and args.loss == "bce":
        valid = (~train_ds.uncertain) if ignore_uncertain else None
        pos_weight = compute_pos_weight(train_ds.labels, valid).to(device)
        print("  Positive-class weighting active:", [round(w, 2) for w in pos_weight.tolist()])

    criterion = build_loss(args.loss, pos_weight=pos_weight).to(device)
    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay, args.layer_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    warmup_steps = max(50, int(steps_per_epoch * args.warmup_epochs))
    total_steps = steps_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps, args.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    eval_every = steps_per_epoch // args.evals_per_epoch if args.evals_per_epoch > 1 else 0
    print(f"  Schedule: total_steps={total_steps:,d} | warmup_steps={warmup_steps:,d} ({args.warmup_epochs} epochs)"
          + (f" | validate every {eval_every:,d} steps" if eval_every else ""))

    # 5. Checkpoints
    best_path = CHECKPOINT_DIR / f"{run_name}_best.pth"
    last_path = CHECKPOINT_DIR / f"{run_name}_last.pth"
    start_epoch = 1
    state = {"best_auc": -1.0, "best_saved": False, "bad_evals": 0}
    history: List[Dict[str, Any]] = []
    patience_evals = args.patience * max(1, args.evals_per_epoch)

    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        state["best_auc"] = ck.get("best_comp_auc", -1.0)
        history = ck.get("history", [])
        state["best_saved"] = best_path.exists()
        print(f"  Resumed from {last_path} at epoch {start_epoch} (Best competition AUC: {state['best_auc']:.4f})")

    def do_eval(epoch_pos: float, train_loss: float) -> bool:
        """Validates on the tune split, keeps the best checkpoint; returns True to stop early."""
        t_eval = time.time()
        metrics, _, _, _ = evaluate(model, tune_loader, criterion, device, use_amp)
        comp_auc = metrics["competition_mean_auc"]
        print(f"\n  [Validation @ epoch {epoch_pos:.2f}] train loss {train_loss:.4f} | val loss {metrics['val_loss']:.4f} "
              f"| {time.time() - t_eval:.0f}s")
        print_auc_summary(metrics, f"Tune Split @ epoch {epoch_pos:.2f}")

        improved = (not math.isnan(comp_auc)) and (comp_auc > state["best_auc"])
        if improved or (not state["best_saved"]):
            state["best_auc"] = comp_auc if improved else state["best_auc"]
            state["best_saved"] = True
            state["bad_evals"] = 0
            torch.save({
                "epoch": epoch_pos,
                "model_name": args.model_name,
                "run_name": run_name,
                "state_dict": model.state_dict(),
                "best_comp_auc": state["best_auc"],
                "metrics": _plain(metrics),
                "args": vars(args),
            }, best_path)
            print(f"  >>> Best model saved: {best_path} (Competition AUC: {comp_auc:.4f})\n")
        else:
            state["bad_evals"] += 1
            print(f"  No improvement over best {state['best_auc']:.4f} ({state['bad_evals']}/{patience_evals or '-'})\n")

        history.append({
            "epoch": round(epoch_pos, 3),
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "comp_auc": comp_auc,
            "comp_auc_mapped": metrics["competition_mean_auc_mapped"],
            "mean_auc_all": metrics["mean_auc_all"],
            "pred_std": metrics["pred_std"],
            "per_class_auc": _plain(metrics["per_class_auc"]),
        })
        return args.patience > 0 and state["bad_evals"] >= patience_evals

    # 6. Epoch Training Loop
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t_start = time.time()
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")

            train_loss, stopped = train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                ignore_uncertain=ignore_uncertain,
                accum_steps=args.accum_steps,
                grad_clip=args.grad_clip,
                epoch=epoch,
                eval_every=eval_every,
                eval_fn=do_eval,
            )
            if not stopped:
                stopped = do_eval(float(epoch), train_loss)

            elapsed = time.time() - t_start
            print(f"  Epoch {epoch} finished in {elapsed / 60:.1f} min | train loss {train_loss:.4f} | best comp AUC {state['best_auc']:.4f}")

            torch.save({
                "epoch": epoch,
                "model_name": args.model_name,
                "run_name": run_name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_comp_auc": state["best_auc"],
                "history": history,
                "args": vars(args),
            }, last_path)

            if stopped:
                print(f"  Early stopping triggered: no improvement for {args.patience} epoch(s) of validations.")
                break
    except torch.cuda.OutOfMemoryError:
        print("\n  !! CUDA out of memory (VRAM cap {:.0%}). Lower --batch_size (and raise --accum_steps to keep the "
              "effective batch), or use --grad_checkpointing. Without the cap, Windows would have silently paged "
              "GPU memory into system RAM and run many times slower.".format(args.max_vram_fraction))
        raise

    # Save training history JSON
    hist_file = OUTPUT_DIR / f"{run_name}_training_history.json"
    with open(hist_file, "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"\nTraining completed. History exported to {hist_file}")

    # 7. Final evaluation of the best checkpoint: held-out test split + official validation set
    final_metrics: Dict[str, Any] = {}
    if not args.skip_test and best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"\n=== Final Evaluation (best checkpoint @ epoch {ck['epoch']}, tune AUC {ck['best_comp_auc']:.4f}) ===")
        loaders = {"test": test_loader}
        loaders["valid"] = create_eval_loader(
            "valid",
            manifest_path=args.manifest,
            chunks=args.chunks,
            batch_size=max(32, args.batch_size),
            num_workers=min(4, args.num_workers),
            image_size=(args.image_size, args.image_size),
            uncertainty_policy=args.uncertainty_policy,
            cache_dir=cache_dir,
            norm=norm,
        )
        titles = {
            "test": "Held-out TEST split (automatic labeler labels)",
            "valid": "Official CheXpert VALID set (radiologist consensus labels)",
        }
        for split, loader in loaders.items():
            if loader is None or len(loader.dataset) == 0:
                continue
            m, y_true, y_pred, unc_arr = evaluate(model, loader, criterion, device, use_amp)
            print_auc_summary(m, titles[split])
            with open(OUTPUT_DIR / f"{run_name}_{split}_metrics.json", "w") as f:
                json.dump(_plain(m), f, indent=2)
            predictions_frame(y_true, y_pred, unc_arr).to_csv(
                OUTPUT_DIR / f"{run_name}_{split}_predictions.csv", index=False)
            final_metrics[split] = m
        print(f"Final metrics & prediction CSVs exported to {OUTPUT_DIR}")

    return {"history": history, "best_comp_auc": state["best_auc"], "final_metrics": final_metrics}


if __name__ == "__main__":
    run_training()

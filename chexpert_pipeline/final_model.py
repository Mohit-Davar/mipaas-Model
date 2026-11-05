"""
final_model.py
=============================================================================
Final deliverable: the 3-model CheXpert ensemble in ONE bundle file.

  DenseNet-121 (512 px) + EfficientNetV2-S (512 px) + RAD-DINO ViT-B/14 (518 px)
  -> each model sees the image and its horizontal flip (TTA) -> probabilities averaged.

Commands
--------
  python final_model.py build                      # pack the 3 best checkpoints -> final_model\\chexpert_ensemble.pth
  python final_model.py calibrate                  # per-class decision thresholds (Youden's J) on the TUNE split
  python final_model.py evaluate --split valid     # AUC + sensitivity/specificity/accuracy (also: test)
  python final_model.py predict --images X --out predictions.csv    # X = image file or folder (png/jpg)

Thresholds are fitted on the tune split ONLY; valid and test are never used to choose anything.
Research use only. Not a medical device and not for clinical decisions.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import ALL_LABELS, BASE_DIR, CHECKPOINT_DIR, COMPETITION_PATHOLOGIES, cache_path_for, pick_cache_dir
from dataset import LetterboxResize, _frontal_mask, load_image, read_manifest
from metrics import compute_chexpert_auc, find_optimal_thresholds
from models import build_model, model_normalization

MEMBERS = ["densenet121", "tf_efficientnetv2_s", "rad_dino"]
FINAL_DIR = BASE_DIR / "final_model"
BUNDLE_PATH = FINAL_DIR / "chexpert_ensemble.pth"
IMAGE_EXT = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
def build_bundle(path: Path = BUNDLE_PATH) -> None:
    members = []
    for name in MEMBERS:
        ck = torch.load(CHECKPOINT_DIR / f"{name}_best.pth", map_location="cpu", weights_only=False)
        a = ck["args"]
        members.append({
            "name": name,
            "model_name": ck["model_name"],
            "image_size": int(a["image_size"]),
            "dropout": a["dropout"],
            "head_hidden": a["head_hidden"],
            "drop_path_rate": a["drop_path_rate"],
            "tune_competition_auc": float(ck["best_comp_auc"]),
            "trained_epoch": float(ck["epoch"]),
            "state_dict": ck["state_dict"],
        })
        print(f"  packed {name:<22s} {a['image_size']}px | best epoch {ck['epoch']:.2f} | tune AUC {ck['best_comp_auc']:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "description": "CheXpert 7-class ensemble: DenseNet-121 + EfficientNetV2-S + RAD-DINO, mean of flip-TTA probabilities",
        "labels": ALL_LABELS,
        "competition_pathologies": COMPETITION_PATHOLOGIES,
        "tta": "hflip",
        "members": members,
        "thresholds": None,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)
    print(f"Bundle written: {path} ({path.stat().st_size / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class _MultiViewDataset(Dataset):
    """Loads each image once and returns one preprocessed tensor per ensemble member."""

    def __init__(self, paths: Sequence[str], transforms_per_member: List[transforms.Compose]):
        self.paths, self.tfs = list(paths), transforms_per_member

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_image(self.paths[i])
        return tuple(tf(img) for tf in self.tfs)


class EnsemblePredictor:
    """Loads the bundle and predicts 7 class probabilities per image (mean of members, flip-TTA)."""

    def __init__(self, bundle_path: Path = BUNDLE_PATH, device: Optional[str] = None, tta: bool = True):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.use_amp = self.device.type == "cuda"
        self.tta = tta
        self.bundle_path = Path(bundle_path)
        b = torch.load(self.bundle_path, map_location="cpu", weights_only=False)
        self.labels: List[str] = b["labels"]
        self.thresholds = b.get("thresholds")
        self.member_info, self.models, self.tfs = [], [], []
        for m in b["members"]:
            net = build_model(
                m["model_name"], num_classes=len(self.labels), pretrained=False, dropout=m["dropout"],
                head_hidden=m["head_hidden"], drop_path_rate=m["drop_path_rate"], image_size=m["image_size"],
            )
            net.load_state_dict(m["state_dict"])
            self.models.append(net.to(self.device).eval())
            mean, std = model_normalization(m["model_name"])
            self.tfs.append(transforms.Compose([
                LetterboxResize(m["image_size"]), transforms.ToTensor(), transforms.Normalize(mean, std)]))
            self.member_info.append({k: v for k, v in m.items() if k != "state_dict"})

    @torch.no_grad()
    def predict(self, paths: Sequence[str], batch_size: int = 16, num_workers: int = 3,
                cache_dir: Optional[Path] = None, per_model: bool = False):
        """Returns probabilities [N, 7] (and per-model [M, N, 7] if per_model). cache_dir: use cached JPEGs."""
        paths = [str(p) for p in paths]
        if cache_dir is not None:
            paths = [str(cache_path_for(p, cache_dir)) if Path(cache_path_for(p, cache_dir)).exists() else p for p in paths]
        loader = DataLoader(_MultiViewDataset(paths, self.tfs), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=self.device.type == "cuda")
        out = [[] for _ in self.models]
        for batch in loader:
            for k, (net, x) in enumerate(zip(self.models, batch)):
                x = x.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    p = torch.sigmoid(net(x).float())
                    if self.tta:
                        p = 0.5 * (p + torch.sigmoid(net(torch.flip(x, dims=[3])).float()))
                out[k].append(p.cpu().numpy())
        stacked = np.stack([np.concatenate(o) for o in out])      # [M, N, 7]
        mean = stacked.mean(axis=0)
        return (mean, stacked) if per_model else mean


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------
def _split_frame(split: str) -> pd.DataFrame:
    df = read_manifest()
    df = df[_frontal_mask(df) & (df["split"] == split)].reset_index(drop=True)
    df = df[df["file_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    return df


def _score_split(pred: EnsemblePredictor, split: str, workers: int):
    df = _split_frame(split)
    raw = df[ALL_LABELS].values.astype(np.float32)
    y, unc = (raw == 1).astype(np.float32), raw == -1
    cache = pick_cache_dir(512)
    t0 = time.time()
    probs = pred.predict(df["file_path"].tolist(), cache_dir=cache, num_workers=workers)
    print(f"  [{split}] {len(df):,} images scored in {time.time() - t0:.0f}s")
    return y, unc, probs


def _operating_point_table(y, unc, probs, thresholds):
    rows = {}
    for c, name in enumerate(ALL_LABELS):
        m = ~unc[:, c]
        yt, flag = y[m, c] == 1, probs[m, c] >= thresholds[name]
        tp, tn = int((flag & yt).sum()), int((~flag & ~yt).sum())
        fp, fn = int((flag & ~yt).sum()), int((~flag & yt).sum())
        rows[name] = {"threshold": thresholds[name], "sensitivity": tp / max(1, tp + fn),
                      "specificity": tn / max(1, tn + fp), "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
                      "n_pos": tp + fn, "n_neg": tn + fp}
    return rows


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_calibrate(args) -> None:
    pred = EnsemblePredictor(args.bundle)
    y, unc, probs = _score_split(pred, "tune", args.workers)
    th = find_optimal_thresholds(y, probs, valid_mask=~unc)
    b = torch.load(args.bundle, map_location="cpu", weights_only=False)
    b["thresholds"] = th
    b["thresholds_fit_on"] = "tune split, certain labels, Youden's J"
    torch.save(b, args.bundle)
    (Path(args.bundle).parent / "thresholds.json").write_text(json.dumps(th, indent=2))
    print("Thresholds (Youden's J on tune split):")
    for k, v in th.items():
        print(f"  {k:<20s} {v:.3f}")


def cmd_evaluate(args) -> None:
    pred = EnsemblePredictor(args.bundle)
    if pred.thresholds is None:
        raise SystemExit("No thresholds in the bundle yet. Run: python final_model.py calibrate")
    y, unc, probs = _score_split(pred, args.split, args.workers)
    auc = compute_chexpert_auc(y, probs, valid_mask=~unc)
    ops = _operating_point_table(y, unc, probs, pred.thresholds)
    print(f"\n=== FINAL ENSEMBLE on {args.split.upper()} (flip TTA) ===")
    print(f"{'class':<20s}{'AUC':>7s}{'thr':>7s}{'sens':>7s}{'spec':>7s}{'acc':>7s}   pos/neg")
    for name in ALL_LABELS:
        o = ops[name]
        star = "*" if name in COMPETITION_PATHOLOGIES else " "
        print(f"{star}{name:<19s}{auc['per_class_auc'][name]:7.3f}{o['threshold']:7.2f}{o['sensitivity']:7.3f}"
              f"{o['specificity']:7.3f}{o['accuracy']:7.3f}   {o['n_pos']}/{o['n_neg']}")
    print(f"\nCompetition mean AUC (5 classes *): {auc['competition_mean_auc']:.4f} | all 7: {auc['mean_auc_all']:.4f}")
    rep = {"split": args.split, "competition_mean_auc": auc["competition_mean_auc"], "mean_auc_all": auc["mean_auc_all"],
           "per_class_auc": auc["per_class_auc"], "operating_points": ops}
    out = FINAL_DIR / f"final_ensemble_{args.split}_report.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"Report saved: {out}")


def _collect_images(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXT)


def cmd_predict(args) -> None:
    files = _collect_images(Path(args.images))
    if not files:
        raise SystemExit(f"No .png/.jpg images found at {args.images}")
    pred = EnsemblePredictor(args.bundle)
    probs = pred.predict([str(f) for f in files], num_workers=args.workers)
    df = pd.DataFrame(probs, columns=[f"prob_{l}" for l in pred.labels])
    if pred.thresholds:
        for i, l in enumerate(pred.labels):
            df[f"flag_{l}"] = (probs[:, i] >= pred.thresholds[l]).astype(int)
    df.insert(0, "image", [str(f) for f in files])
    df.to_csv(args.out, index=False)
    print(f"{len(df)} image(s) -> {args.out}")
    print(df.round(3).head(10).to_string(index=False))
    print("\nResearch use only. Not a medical device; not for clinical decisions.")


def main() -> None:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description="CheXpert final ensemble")
    p.add_argument("command", choices=["build", "calibrate", "evaluate", "predict"])
    p.add_argument("--bundle", type=Path, default=BUNDLE_PATH)
    p.add_argument("--split", choices=["tune", "test", "valid"], default="valid")
    p.add_argument("--images", type=str)
    p.add_argument("--out", type=str, default="predictions.csv")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    if args.command == "build":
        build_bundle(args.bundle)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        if not args.images:
            raise SystemExit("--images is required for predict")
        cmd_predict(args)


if __name__ == "__main__":
    main()

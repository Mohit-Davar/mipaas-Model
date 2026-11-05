"""
label_sanity_probe.py
=============================================================================
Quick check that the labels actually describe the images, before spending GPU hours.

Fits a logistic regression per class on frozen ImageNet DenseNet-121 features
(no fine-tuning) using a few thousand train images, and scores it on the tune split.
  * Labels aligned with images : mean AUC ~0.64 (measured on the real CheXpert labels)
  * Labels scrambled vs images : mean AUC ~0.50 (the CheXbert JSON manifest scored 0.50)

Exit code 0 = PASS (mean AUC >= --min_auc), 1 = FAIL.
Runtime: ~1 min with the image cache, ~5-8 min reading full-res PNGs from the HDD.

Usage:  python tools\\label_sanity_probe.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALL_LABELS, MANIFEST_PARQUET, cache_path_for, pick_cache_dir  # noqa: E402
from dataset import _frontal_mask, get_transforms, load_image  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Linear-probe check that labels match images")
    p.add_argument("--manifest", type=str, default=str(MANIFEST_PARQUET))
    p.add_argument("--cache_dir", type=str, default="auto")
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_eval", type=int, default=1500)
    p.add_argument("--image_size", type=int, default=320)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min_auc", type=float, default=0.57)
    return p.parse_args()


class _Images(Dataset):
    def __init__(self, paths, tf, cache_dir):
        self.paths, self.tf = list(paths), tf
        use_cache = bool(cache_dir) and Path(cache_dir).is_dir()
        self.cache = [str(cache_path_for(p, cache_dir)) for p in self.paths] if use_cache else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.cache is not None:
            try:
                return self.tf(load_image(self.cache[i]))
            except FileNotFoundError:
                pass
        return self.tf(load_image(self.paths[i]))


@torch.no_grad()
def extract(model, paths, tf, cache_dir, workers, device) -> np.ndarray:
    loader = DataLoader(_Images(paths, tf, cache_dir), batch_size=64, num_workers=workers)
    feats = []
    for x in loader:
        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            feats.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(feats).numpy()


def main():
    args = parse_args()
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(args.manifest)
    df = df[_frontal_mask(df)]
    tr = df[df["split"] == "train"]
    ev = df[df["split"] == "tune"]
    tr = tr.sample(min(args.n_train, len(tr)), random_state=0)
    ev = ev.sample(min(args.n_eval, len(ev)), random_state=0)
    print(f"[probe] manifest={args.manifest}\n[probe] {len(tr):,} train / {len(ev):,} tune frontal images, "
          f"{args.image_size}px, frozen ImageNet densenet121 features")

    if args.cache_dir == "auto":
        args.cache_dir = pick_cache_dir(args.image_size)
    _, tf = get_transforms((args.image_size, args.image_size), hflip=False, rotation=0, jitter=0)
    model = timm.create_model("densenet121", pretrained=True, num_classes=0).to(device).eval()
    x_tr = extract(model, tr["file_path"], tf, args.cache_dir, args.workers, device)
    x_ev = extract(model, ev["file_path"], tf, args.cache_dir, args.workers, device)
    scaler = StandardScaler().fit(x_tr)
    x_tr, x_ev = scaler.transform(x_tr), scaler.transform(x_ev)

    aucs = {}
    print(f"\n  {'label':<18s} {'probe AUC':>9s}   tune pos/neg (certain labels)")
    for lbl in ALL_LABELS:
        y_tr = (tr[lbl].values != 0).astype(int)  # uncertain counted as positive for the probe
        keep = ev[lbl].values != -1
        y_ev = (ev[lbl].values[keep] == 1).astype(int)
        if y_tr.min() == y_tr.max() or y_ev.min() == y_ev.max():
            print(f"  {lbl:<18s} {'n/a':>9s}")
            continue
        clf = LogisticRegression(C=0.05, max_iter=3000).fit(x_tr, y_tr)
        aucs[lbl] = roc_auc_score(y_ev, clf.predict_proba(x_ev[keep])[:, 1])
        print(f"  {lbl:<18s} {aucs[lbl]:>9.4f}   {y_ev.sum()}/{(1 - y_ev).sum()}")

    mean_auc = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    passed = mean_auc >= args.min_auc
    print(f"\n  Mean probe AUC: {mean_auc:.4f}  (threshold {args.min_auc:.2f}) -> {'PASS' if passed else 'FAIL'}"
          f"  [{time.time() - t0:.0f}s]")
    if not passed:
        print("  !! Labels carry (almost) no information about the images. Do NOT train on this manifest.\n"
              "  !! Rebuild it from the original CheXpert labels: python tools\\build_manifest_chexpert.py")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

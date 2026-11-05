"""
build_image_cache.py
=============================================================================
One-time pre-resize of the CheXpert Plus PNGs into a fast cache:
long side 512 px, 8-bit grayscale JPEG (quality 95), mirrored folder layout.

Why
---
The originals are 2000-3400 px PNGs of 3-9 MB on a SATA hard drive (D:). Reading and
decoding them every epoch capped training at ~5-20 img/s (2 epochs took ~18 h), while
the GPU can train a single backbone at ~130-210 img/s. From the cache, 8 loader workers
deliver far more than the GPU needs. Training at 384 px loses nothing by reading
512 px sources.

Caches are stored per size as C:\\chexpert_cache\\jpg<max_side> (NVMe drive). Training picks
the smallest cache whose side >= --image_size and refuses a cache that is too small
(that would upscale). Files missing from the cache are read from the original PNG.

Usage
-----
  python tools\\build_image_cache.py                      # 512 px  -> train at <= 512 (default 384)
  python tools\\build_image_cache.py --max_side 1024      # 1024 px -> train at up to 1024 (e.g. 768)
  python tools\\build_image_cache.py --limit 500          # quick speed test
Re-running resumes: existing cache files are skipped.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_ROOT, MANIFEST_PARQUET, cache_path_for  # noqa: E402  (config does not import torch)

EST_BYTES_PER_IMAGE = 75_000  # measured 70 KB average for 512 px grayscale JPEG q95, plus margin


def parse_args():
    p = argparse.ArgumentParser(description="Pre-resize CheXpert Plus PNGs into a fast JPEG cache")
    p.add_argument("--manifest", type=str, default=str(MANIFEST_PARQUET))
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Default: C:\\chexpert_cache\\jpg<max_side> (found automatically by training)")
    p.add_argument("--max_side", type=int, default=512,
                   help="Long side of cached images; must be >= the training --image_size (e.g. 1024 for 768px training)")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--all_views", action="store_true", help="Also cache lateral views (the pipeline trains on frontal only)")
    p.add_argument("--splits", nargs="+", default=None, help="Restrict to these splits (default: all)")
    p.add_argument("--limit", type=int, default=None, help="Process at most N images (speed test)")
    return p.parse_args()


def _convert(task):
    src, dst, max_side, quality = task
    try:
        with Image.open(src) as im:
            if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                arr = np.asarray(im, dtype=np.float32)
                lo, hi = float(arr.min()), float(arr.max())
                img = Image.fromarray(((arr - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8))
            else:
                img = im.convert("L")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS, reducing_gap=3.0)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        img.save(tmp, format="JPEG", quality=quality)
        os.replace(tmp, dst)  # atomic: an interrupted run never leaves a truncated cache file
        return os.path.getsize(dst), None
    except Exception as e:  # noqa: BLE001 - report and continue
        return 0, f"{src}: {e!r}"


def main():
    args = parse_args()
    import pandas as pd

    df = pd.read_parquet(args.manifest)
    if not args.all_views:
        df = df[df["is_frontal"].astype(bool)]
    if args.splits:
        df = df[df["split"].isin(args.splits)]
    sources = sorted(set(df["file_path"].astype(str)))  # path order ~ on-disk order (HDD friendly)

    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_ROOT / f"jpg{args.max_side}"
    tasks = []
    for src in sources:
        dst = str(cache_path_for(src, cache_dir))
        if not os.path.exists(dst):
            tasks.append((src, dst, args.max_side, args.quality))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    cache_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_dir).free
    need = len(tasks) * EST_BYTES_PER_IMAGE
    print("=" * 72)
    print(" CheXpert Plus image cache")
    print(f"  cache dir : {cache_dir}  ({max(free, 0) / 1e9:.0f} GB free)")
    print(f"  images    : {len(sources):,} selected | {len(sources) - len(tasks):,} already cached | {len(tasks):,} to do")
    print(f"  format    : JPEG q{args.quality}, long side {args.max_side}px, grayscale | ~{need / 1e9:.1f} GB to write")
    print("=" * 72)
    if need * 1.2 > free:
        raise SystemExit(f"[ERROR] Not enough free space in {cache_dir} (need ~{need * 1.2 / 1e9:.0f} GB).")
    if not tasks:
        print("Nothing to do.")
        return

    t0 = time.time()
    written, errors = 0, []
    report_every = max(100, min(2000, len(tasks) // 50))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (size, err) in enumerate(ex.map(_convert, tasks, chunksize=8), 1):
            written += size
            if err:
                errors.append(err)
            if i % report_every == 0 or i == len(tasks):
                el = time.time() - t0
                rate = i / max(el, 1e-6)
                eta = (len(tasks) - i) / max(rate, 1e-6) / 60
                print(f"  {i:>7,d}/{len(tasks):,} | {rate:5.1f} img/s | {written / max(1, i - len(errors)) / 1e3:5.1f} KB/img "
                      f"| {written / 1e9:5.2f} GB | ETA {eta:5.1f} min | errors {len(errors)}", flush=True)

    print(f"\nDone: {len(tasks) - len(errors):,} images cached in {(time.time() - t0) / 60:.1f} min "
          f"({written / 1e9:.2f} GB). Errors: {len(errors)}")
    for e in errors[:10]:
        print("   ", e)
    if cache_dir.parent.resolve() != CACHE_ROOT.resolve() or cache_dir.name.lower() != f"jpg{args.max_side}":
        print(f"\nCustom cache location: pass --cache_dir {cache_dir} to train.py/evaluate.py "
              f"(automatic selection only scans {CACHE_ROOT}\\jpg<side>).")


if __name__ == "__main__":
    main()

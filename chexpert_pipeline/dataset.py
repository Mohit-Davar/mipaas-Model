"""
dataset.py
=============================================================================
Standardized CheXpert Plus Dataset and DataLoader Pipeline.
=============================================================================
Features:
- Aspect-ratio preserving isotropic LetterboxResize (prevents CTR distortion).
- Robust image decoder supporting 8-bit, 16-bit, and float grayscale/RGB radiography.
- Reads from the pre-resized 512 px cache (tools/build_image_cache.py) when present,
  falling back to the original PNGs per image.
- CheXpert uncertainty handling: u_ones, u_zeros, u_ignore, or per-class 'mixed',
  with optional label smoothing of uncertain targets (training split only).
- Strict patient-level isolation verification (zero train/tune/test/valid leakage).
- Multi-worker loading (Windows spawn-safe; entry scripts use a __main__ guard).
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import (
    ALL_LABELS,
    CACHE_ROOT,
    CONFIG,
    DATASET_DIR,
    LSR_RANGES,
    MANIFEST_PARQUET,
    MANIFEST_CSV,
    IMAGENET_MEAN,
    IMAGENET_STD,
    UNCERTAINTY_PER_CLASS,
    cache_path_for,
    pick_cache_dir,
)

SPLITS = ("train", "tune", "test", "valid")
CACHE_UPSCALE_TOLERANCE = 0.98  # a cache may be up to 2% smaller than the training size (512 -> 518)


def _as_size(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Normalizes size input to an (H, W) tuple."""
    if isinstance(size, int):
        return (size, size)
    size = tuple(size)
    return size if len(size) == 2 else (size[0], size[0])


class LetterboxResize:
    """
    Isotropic aspect-ratio preserving resize with symmetric canvas padding.
    Crucial for chest radiography: isotropic scaling prevents artificial
    dilation or compression of the Cardiothoracic Ratio (CTR).
    """

    def __init__(self, target_size: Union[int, Tuple[int, int]] = (384, 384), fill: int = 0):
        self.target_size = _as_size(target_size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        tw, th = self.target_size
        scale = min(tw / w, th / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (tw, th), (self.fill,) * 3)
        canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
        return canvas


def load_image(path: Union[str, Path]) -> Image.Image:
    """
    Opens an image as standard 8-bit RGB.
    Robustly handles high-dynamic-range 16-bit ('I;16', 'I') and float ('F') radiographs.
    """
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))
    return img.convert("RGB")


def process_labels(
    raw: np.ndarray,
    policy: str = "u_ones",
    labels: List[str] = ALL_LABELS,
    smooth: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translates raw CheXpert labels {-1, 0, 1, NaN} into training targets and an uncertainty mask.
    Policies:
      - 'u_ones':   Uncertain (-1) -> 1.0 for every class
      - 'u_zeros':  Uncertain (-1) -> 0.0 for every class
      - 'u_ignore': Uncertain (-1) -> 0.0 and masked out of the loss
      - 'mixed':    Per-class choice from config.UNCERTAINTY_PER_CLASS
    smooth=True replaces the hard 1/0 of uncertain entries by a fixed random value drawn
    from config.LSR_RANGES ('ones' -> 0.55-0.85, 'zeros' -> 0-0.30). Use for training only.
    """
    raw = np.nan_to_num(raw.astype(np.float32), nan=0.0)
    bad = ~np.isin(raw, (-1.0, 0.0, 1.0))
    if bad.any():
        raise ValueError(f"Labels must be in {{-1, 0, 1}}; found unexpected values: {np.unique(raw[bad])[:5]}")

    if policy == "u_ones":
        modes = ["ones"] * len(labels)
    elif policy in ("u_zeros", "u_ignore"):
        modes = ["zeros"] * len(labels)
    elif policy == "mixed":
        modes = [UNCERTAINTY_PER_CLASS.get(name, "ones") for name in labels]
    else:
        raise ValueError(f"Unknown uncertainty_policy '{policy}'. Choose 'mixed', 'u_ones', 'u_zeros', or 'u_ignore'.")

    uncertain = (raw == -1.0)
    targets = raw.copy()
    rng = np.random.default_rng(seed)
    for c, mode in enumerate(modes):
        u = uncertain[:, c]
        if not u.any():
            continue
        if smooth and policy != "u_ignore":
            lo, hi = LSR_RANGES[mode]
            targets[u, c] = rng.uniform(lo, hi, size=int(u.sum()))
        else:
            targets[u, c] = 1.0 if mode == "ones" else 0.0

    return targets.astype(np.float32), uncertain


class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for multi-label CheXpert Plus radiographs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        uncertainty_policy: str = "u_ones",
        dataset_dir: Path = DATASET_DIR,
        max_retries: int = 3,
        label_smoothing: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.uncertainty_policy = uncertainty_policy
        self.max_retries = max_retries

        # Resolve paths to absolute paths
        paths = []
        for p in self.df["file_path"].tolist():
            p_obj = Path(str(p))
            if p_obj.is_absolute() and p_obj.exists():
                paths.append(str(p_obj))
            else:
                paths.append(str(dataset_dir / p_obj))
        self.file_paths = paths

        # Pre-resized cache (optional). Missing cache files fall back to the original PNG.
        self.cache_paths = None
        if cache_dir is not None and Path(cache_dir).is_dir():
            self.cache_paths = [str(cache_path_for(p, cache_dir)) for p in self.file_paths]

        # Process labels and uncertainty masks
        labels, uncertain = process_labels(
            self.df[ALL_LABELS].values, uncertainty_policy, smooth=label_smoothing, seed=seed
        )
        self.labels = torch.from_numpy(labels)       # [N, num_classes] float32
        self.uncertain = torch.from_numpy(uncertain) # [N, num_classes] bool

    def __len__(self) -> int:
        return len(self.df)

    def _read(self, i: int) -> Image.Image:
        if self.cache_paths is not None:
            try:
                return load_image(self.cache_paths[i])
            except FileNotFoundError:
                pass
        return load_image(self.file_paths[i])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last_err = None
        for attempt in range(self.max_retries + 1):
            target_idx = (idx + attempt) % len(self)
            try:
                image = self._read(target_idx)
                break
            except Exception as e:
                last_err = e
                warnings.warn(f"Could not read image '{self.file_paths[target_idx]}': {e}")
        else:
            raise RuntimeError(
                f"{self.max_retries + 1} consecutive unreadable images starting at '{self.file_paths[idx]}': {last_err}"
            )

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[target_idx], self.uncertain[target_idx]


def get_transforms(
    image_size: Union[int, Tuple[int, int]] = (384, 384),
    hflip: bool = True,
    rotation: float = 7.0,
    jitter: float = 0.1,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform).
    Both use isotropic LetterboxResize followed by the model's normalization
    (ImageNet statistics by default; RAD-DINO uses its own, see models.model_normalization).
    """
    image_size = _as_size(image_size)

    train_ops = [LetterboxResize(image_size)]
    if hflip:
        train_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if rotation > 0:
        train_ops.append(transforms.RandomRotation(degrees=rotation, interpolation=InterpolationMode.BILINEAR))
    if jitter > 0:
        train_ops.append(transforms.ColorJitter(brightness=jitter, contrast=jitter))
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    val_ops = [
        LetterboxResize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def _frontal_mask(df: pd.DataFrame) -> pd.Series:
    """Extracts boolean mask identifying frontal (AP/PA) projections."""
    if "is_frontal" in df.columns:
        col = df["is_frontal"]
        if col.dtype == bool:
            return col
        return col.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "frontal"})
    if "Frontal/Lateral" in df.columns:
        return df["Frontal/Lateral"].astype(str).str.strip().str.lower() == "frontal"
    raise KeyError("Manifest must contain 'is_frontal' or 'Frontal/Lateral' column.")


def _check_patient_leakage(splits: Dict[str, pd.DataFrame]) -> None:
    """Verifies that no patient appears in more than one partition."""
    patient_col = None
    for candidate in ("patient_id", "patient", "Patient ID", "subject_id"):
        if any(candidate in d.columns for d in splits.values() if len(d)):
            patient_col = candidate
            break

    if patient_col is None:
        warnings.warn("No patient identifier column found in manifest; cannot formally verify split independence.")
        return

    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a_name, b_name = split_names[i], split_names[j]
            a_df, b_df = splits[a_name], splits[b_name]
            if len(a_df) and len(b_df):
                overlap = set(a_df[patient_col]).intersection(set(b_df[patient_col]))
                if overlap:
                    raise RuntimeError(
                        f"PATIENT LEAKAGE VIOLATION: {len(overlap)} patients appear in both '{a_name}' and '{b_name}'!"
                    )
    print(f"[dataset] Patient isolation verified: 0 leakage across splits (checked '{patient_col}').")


def _preflight_paths(df: pd.DataFrame, split_name: str, dataset_dir: Path, sample_n: int = 128, max_missing: float = 0.02) -> None:
    """Performs quick filesystem existence check on sample of paths."""
    if len(df) == 0:
        return
    sample_paths = df["file_path"].sample(n=min(sample_n, len(df)), random_state=42).tolist()
    missing = 0
    for p in sample_paths:
        p_obj = Path(str(p))
        if not (p_obj.is_absolute() and p_obj.exists()) and not (dataset_dir / p_obj).exists():
            missing += 1

    missing_frac = missing / len(sample_paths)
    if missing_frac > max_missing:
        raise FileNotFoundError(
            f"[{split_name}] {missing}/{len(sample_paths)} sampled paths missing. Example: {sample_paths[0]}. "
            f"Please verify CHEXPERT_DATA_DIR ({dataset_dir})."
        )


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


AVAILABLE_CHUNKS: Dict[Union[int, str], str] = {
    0: "png_chexpert_plus_chunk_0",
    "0": "png_chexpert_plus_chunk_0",
    "chunk_0": "png_chexpert_plus_chunk_0",
    1: "png_chexpert_plus_chunk_1",
    "1": "png_chexpert_plus_chunk_1",
    "chunk_1": "png_chexpert_plus_chunk_1",
    2: "png_chexpert_plus_chunk_2",
    "2": "png_chexpert_plus_chunk_2",
    "chunk_2": "png_chexpert_plus_chunk_2",
    3: "png_chexpert_plus_chunk_3",
    "3": "png_chexpert_plus_chunk_3",
    "chunk_3": "png_chexpert_plus_chunk_3",
    4: "png_chexpert_plus_chunk_4_3",
    "4": "png_chexpert_plus_chunk_4_3",
    "4_3": "png_chexpert_plus_chunk_4_3",
    "chunk_4": "png_chexpert_plus_chunk_4_3",
    "chunk_4_3": "png_chexpert_plus_chunk_4_3",
}


def load_chunks_manually(
    df: pd.DataFrame,
    dataset_dir: Path,
    chunk_ids: Optional[List[Union[int, str]]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Explicitly and manually imports dataset chunks (0, 1, 2, 3, 4).
    Audits disk folders and matches rows per chunk folder.
    """
    if chunk_ids is None:
        chunk_ids = [0, 1, 2, 3, 4]

    resolved_chunks = []
    for cid in chunk_ids:
        c_str = str(cid).strip().lower()
        if c_str in AVAILABLE_CHUNKS:
            folder = AVAILABLE_CHUNKS[c_str]
        elif cid in AVAILABLE_CHUNKS:
            folder = AVAILABLE_CHUNKS[cid]
        else:
            folder = f"png_chexpert_plus_chunk_{cid}"
        if folder not in resolved_chunks:
            resolved_chunks.append(folder)

    if verbose:
        print("\n" + "=" * 68)
        print(" [Manual Chunk Loader] Importing Specified Dataset Chunks...")
        print("=" * 68)

    chunk_dfs = []
    for folder in resolved_chunks:
        chunk_dir = dataset_dir / folder
        status = "EXISTS" if chunk_dir.exists() else "MISSING"

        # Match file_path containing the specific chunk folder
        mask = df["file_path"].astype(str).str.contains(folder, regex=False)
        sub_df = df[mask].reset_index(drop=True)

        if verbose:
            print(f"  --> Importing: '{folder}'")
            print(f"      Disk Folder: {chunk_dir} [{status}]")
            print(f"      Images loaded from this chunk: {len(sub_df):,d} records")

        if len(sub_df) == 0:
            warnings.warn(f"Chunk '{folder}' produced 0 records!")
        else:
            chunk_dfs.append(sub_df)

    if not chunk_dfs:
        raise ValueError(f"No images loaded from requested chunks: {chunk_ids}")

    combined = pd.concat(chunk_dfs, ignore_index=True)
    if verbose:
        print(f"\n  >>> Total images manually merged across {len(chunk_dfs)} chunks: {len(combined):,d}")
        print("=" * 68 + "\n")
    return combined


def read_manifest(manifest_path: Union[str, Path] = MANIFEST_PARQUET) -> pd.DataFrame:
    """Reads the split manifest (parquet preferred, CSV fallback)."""
    manifest_path = Path(manifest_path)
    if manifest_path.suffix == ".parquet" and manifest_path.exists():
        return pd.read_parquet(manifest_path)
    if manifest_path.suffix == ".csv" and manifest_path.exists():
        return pd.read_csv(manifest_path)
    if MANIFEST_PARQUET.exists():
        return pd.read_parquet(MANIFEST_PARQUET)
    if MANIFEST_CSV.exists():
        return pd.read_csv(MANIFEST_CSV)
    raise FileNotFoundError(f"Manifest not found at {manifest_path}")


def _loader_kwargs(num_workers: int, persistent: bool) -> Dict:
    kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(persistent and num_workers > 0),
        worker_init_fn=(_seed_worker if num_workers > 0 else None),
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    return kw


def _cache_side(cache_dir: Path, n: int = 5) -> Optional[int]:
    """Long side (px) of a few cached images; None if the cache is still empty."""
    sides = []
    for f in cache_dir.rglob("*.jpg"):
        with Image.open(f) as im:
            sides.append(max(im.size))
        if len(sides) >= n:
            break
    return max(sides) if sides else None


def _cache_status(cache_dir: Optional[Union[str, Path]],
                  image_size: Union[int, Tuple[int, int]] = CONFIG["image_size"]) -> Optional[Path]:
    """Resolves the image cache for a training resolution ('auto' = smallest cache that is large enough)."""
    if cache_dir is None or str(cache_dir).strip() == "":
        return None
    need = max(_as_size(image_size))
    if str(cache_dir).lower() == "auto":
        cache_dir = pick_cache_dir(need)
        if cache_dir is None:
            print(f"[dataset] No image cache with long side >= {need}px under {CACHE_ROOT} -> reading "
                  f"full-resolution PNGs (slow on a hard drive). Build one: "
                  f"python tools\\build_image_cache.py --max_side {max(512, need)}")
            return None
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        print(f"[dataset] No image cache at {cache_dir} -> reading full-resolution PNGs "
              f"(slow on a hard drive; build it with tools/build_image_cache.py).")
        return None
    side = _cache_side(cache_dir)
    if side is not None and side < CACHE_UPSCALE_TOLERANCE * need:  # e.g. 512 cache serves 518 (RAD-DINO)
        raise ValueError(
            f"Image cache {cache_dir} holds {side}px images but training/evaluation runs at {need}px: "
            f"images would be UPSCALED and the extra resolution wasted. Build a larger cache first: "
            f"python tools\\build_image_cache.py --max_side {max(1024, need)}"
        )
    print(f"[dataset] Using pre-resized image cache: {cache_dir}" + (f" ({side}px)" if side else ""))
    return cache_dir


def create_dataloaders(
    manifest_path: Union[str, Path] = MANIFEST_PARQUET,
    dataset_dir: Union[str, Path] = DATASET_DIR,
    chunks: Optional[List[Union[int, str]]] = None,
    batch_size: int = CONFIG["batch_size"],
    num_workers: int = CONFIG["num_workers"],
    image_size: Union[int, Tuple[int, int]] = CONFIG["image_size"],
    frontal_only: bool = CONFIG["frontal_only"],
    uncertainty_policy: str = CONFIG["uncertainty_policy"],
    sample_limit: Optional[int] = None,
    tune_sample_limit: Optional[int] = CONFIG.get("tune_sample_limit", 5000),
    seed: int = CONFIG["seed"],
    hflip: bool = CONFIG["hflip"],
    rotation: float = CONFIG.get("rotation", 7.0),
    jitter: float = CONFIG.get("jitter", 0.1),
    check_paths: bool = True,
    label_smoothing: bool = CONFIG.get("label_smoothing", False),
    cache_dir: Optional[Union[str, Path]] = "auto",
    norm: Optional[Tuple[List[float], List[float]]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function creating (train_loader, tune_loader, test_loader).
    Supports manual loading of chunks (chunks=[0, 1, 2, 3, 4]).
    The official CheXpert validation set (split == 'valid') is excluded here;
    use create_eval_loader('valid') for it.
    """
    dataset_dir = Path(dataset_dir)
    df = read_manifest(manifest_path)

    # Manually import specified chunks (chunks 0, 1, 2, 3, 4)
    if chunks is None:
        chunks = CONFIG.get("chunks", [0, 1, 2, 3, 4])
    df = load_chunks_manually(df, dataset_dir=dataset_dir, chunk_ids=chunks)

    if frontal_only:
        df = df[_frontal_mask(df)].reset_index(drop=True)

    # Split into train, tune, test (+ valid for the leakage check) partitions
    all_splits = {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}
    for s in ("train", "tune"):
        if len(all_splits[s]) == 0:
            raise ValueError(f"Partition '{s}' is empty after filtering (check split column).")

    _check_patient_leakage(all_splits)
    splits = {s: all_splits[s] for s in ("train", "tune", "test")}

    # Audit and print verified multi-chunk representation
    def _chunk_tag(p):
        for c in ("chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_4_3"):
            if c in str(p):
                return f"png_chexpert_plus_{c}"
        return "other"

    chunk_counts = splits["train"]["file_path"].apply(_chunk_tag).value_counts().to_dict()
    total_train = len(splits["train"])
    print(f"[dataset] Verified representation across all 5 chunks in training split ({total_train:,} total images):")
    for chunk_name in sorted(chunk_counts.keys()):
        count = chunk_counts[chunk_name]
        pct = (count / max(1, total_train)) * 100
        print(f"  * {chunk_name:<28s}: {count:>6,d} images ({pct:>5.1f}%)")

    if sample_limit is not None:
        lim = {
            "train": sample_limit,
            "tune": max(16, sample_limit // 4),
            "test": max(16, sample_limit // 4),
        }
        splits = {
            s: d.sample(n=min(len(d), lim[s]), random_state=seed).reset_index(drop=True)
            for s, d in splits.items()
        }
    elif tune_sample_limit and len(splits["tune"]) > tune_sample_limit:
        splits["tune"] = splits["tune"].sample(n=tune_sample_limit, random_state=seed).reset_index(drop=True)
        print(f"[dataset] Subsampled tune split to {len(splits['tune']):,} images for fast intermediate validation.")

    if check_paths:
        for s, d in splits.items():
            _preflight_paths(d, s, dataset_dir)

    cache_dir = _cache_status(cache_dir, image_size)
    mean, std = norm or (IMAGENET_MEAN, IMAGENET_STD)
    train_tf, val_tf = get_transforms(image_size, hflip=hflip, rotation=rotation, jitter=jitter, mean=mean, std=std)
    train_ds = CheXpertDataset(splits["train"], train_tf, uncertainty_policy, dataset_dir=dataset_dir,
                               label_smoothing=label_smoothing, cache_dir=cache_dir, seed=seed)
    tune_ds = CheXpertDataset(splits["tune"], val_tf, uncertainty_policy, dataset_dir=dataset_dir, cache_dir=cache_dir)
    test_ds = CheXpertDataset(splits["test"], val_tf, uncertainty_policy, dataset_dir=dataset_dir, cache_dir=cache_dir)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_ds) > batch_size),
        generator=g,
        **_loader_kwargs(num_workers, persistent=True),
    )

    # Eval loaders: batch >= 32 (no backward graph); fewer workers to save RAM
    eval_bs = max(1, min(len(tune_ds), max(batch_size, 32)))
    eval_workers = min(num_workers, 4)
    tune_loader = DataLoader(tune_ds, batch_size=eval_bs, shuffle=False, drop_last=False,
                             **_loader_kwargs(eval_workers, persistent=True))
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, drop_last=False,
                             **_loader_kwargs(eval_workers, persistent=False))

    print(f"[dataset] Dataloaders ready: train={len(train_ds):,} | tune={len(tune_ds):,} | test={len(test_ds):,} "
          f"| policy={uncertainty_policy}{' +LSR' if label_smoothing else ''} | size={_as_size(image_size)} "
          f"| workers={num_workers}")

    return train_loader, tune_loader, test_loader


def create_eval_loader(
    split: str,
    manifest_path: Union[str, Path] = MANIFEST_PARQUET,
    dataset_dir: Union[str, Path] = DATASET_DIR,
    chunks: Optional[List[Union[int, str]]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: Union[int, Tuple[int, int]] = CONFIG["image_size"],
    frontal_only: bool = CONFIG["frontal_only"],
    uncertainty_policy: str = CONFIG["uncertainty_policy"],
    sample_limit: Optional[int] = None,
    seed: int = CONFIG["seed"],
    cache_dir: Optional[Union[str, Path]] = "auto",
    norm: Optional[Tuple[List[float], List[float]]] = None,
) -> Optional[DataLoader]:
    """
    Full (un-subsampled) evaluation loader for one split: 'tune', 'test', or 'valid'
    (the official CheXpert validation set with radiologist labels, when the manifest has it).
    Returns None when the split has no rows.
    """
    dataset_dir = Path(dataset_dir)
    df = read_manifest(manifest_path)
    df = load_chunks_manually(df, dataset_dir=dataset_dir,
                              chunk_ids=chunks or CONFIG.get("chunks", [0, 1, 2, 3, 4]), verbose=False)
    if frontal_only:
        df = df[_frontal_mask(df)]
    df = df[df["split"] == split].reset_index(drop=True)
    if len(df) == 0:
        return None
    if sample_limit is not None and len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=seed).reset_index(drop=True)

    mean, std = norm or (IMAGENET_MEAN, IMAGENET_STD)
    _, val_tf = get_transforms(image_size, mean=mean, std=std)
    ds = CheXpertDataset(df, val_tf, uncertainty_policy, dataset_dir=dataset_dir,
                         cache_dir=_cache_status(cache_dir, image_size))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False,
                      **_loader_kwargs(num_workers, persistent=False))

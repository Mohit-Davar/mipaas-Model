"""
build_manifest_chexpert.py
=============================================================================
Rebuilds split_manifest.{parquet,csv} from the ORIGINAL CheXpert v1.0 labels
(train.csv / valid.csv), matched to the CheXpert Plus PNG chunks on disk.

Why this exists
---------------
The CheXbert label files shipped with CheXpert Plus (report_fixed.json,
findings_fixed.json, impression_fixed.json), which tools/build_manifest.py used,
do not line up with their own `path_to_image` column:
  * The frontal and lateral images of one study come from ONE report, so they
    must carry identical labels. In report_fixed.json they differ in 92% of
    multi-view studies (identical in only 7.6%).
  * A logistic-regression probe on frozen ImageNet DenseNet-121 features scores
    AUC ~0.50 on all 7 target classes (correctly labelled CheXpert: ~0.70-0.85).
  * Every model trained on them stayed at AUC ~0.50 (DenseNet-121, TriBackboneFusion).
See also https://github.com/Stanford-AIMI/chexpert-plus/issues/13 (identical report
text with opposite CheXbert labels).

Inputs
------
  --train_csv  CheXpert v1.0 train.csv  (from CheXpert-v1.0 or CheXpert-v1.0-small,
               Stanford AIMI, same research-use agreement as CheXpert Plus)
  --valid_csv  CheXpert v1.0 valid.csv  (234 images, radiologist consensus = gold standard)

Output
------
  split_manifest.parquet / .csv with the schema the pipeline expects plus
  sex, age, ap_pa, label_source. Splits:
    train / tune / test : patient-level, stratified, from train.csv (default 90/5/5)
    valid               : the official CheXpert validation set (final benchmark)
  The previous (misaligned) manifest is kept as split_manifest.chexbert_misaligned.*

Usage
-----
  python tools\\build_manifest_chexpert.py
  python tools\\build_manifest_chexpert.py --train_csv D:\\path\\train.csv --valid_csv D:\\path\\valid.csv
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    BASE_DIR,
    COMPETITION_PATHOLOGIES,
    DATASET_DIR,
    FULL_14_LABELS,
    LABELS_DIR,
    MANIFEST_CSV,
    MANIFEST_PARQUET,
)

CHUNK_DIRS = [
    "png_chexpert_plus_chunk_0",
    "png_chexpert_plus_chunk_1",
    "png_chexpert_plus_chunk_2",
    "png_chexpert_plus_chunk_3",
    "png_chexpert_plus_chunk_4_3",
]
PATH_PATTERN = r"(train|valid)/(patient\d+)/(study\d+)/(view\d+_(?:frontal|lateral))\.(?:jpg|jpeg|png)$"


def parse_args():
    p = argparse.ArgumentParser(description="Build split manifest from original CheXpert v1.0 labels")
    p.add_argument("--train_csv", type=str, default=str(LABELS_DIR / "train.csv"))
    p.add_argument("--valid_csv", type=str, default=str(LABELS_DIR / "valid.csv"))
    p.add_argument("--dataset_dir", type=str, default=str(DATASET_DIR))
    p.add_argument("--out_dir", type=str, default=str(BASE_DIR))
    p.add_argument("--tune_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def index_patients(dataset_dir: Path) -> dict:
    """(subfolder, patient) -> list of chunk PNG roots containing that patient folder."""
    idx = defaultdict(list)
    for chunk in CHUNK_DIRS:
        root = dataset_dir / chunk / "PNG"
        for sub in ("train", "valid"):
            d = root / sub
            if d.is_dir():
                with os.scandir(d) as it:
                    for entry in it:
                        if entry.is_dir():
                            idx[(sub, entry.name)].append(root)
    return idx


def read_label_csv(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"\n[ERROR] Label file not found: {path}\n"
            f"  Download CheXpert v1.0 'train.csv' and 'valid.csv' from Stanford AIMI\n"
            f"  (they ship with CheXpert-v1.0 and CheXpert-v1.0-small) and place them in:\n"
            f"  {LABELS_DIR}\n"
        )
    df = pd.read_csv(path)
    missing = ({"Path"} | set(FULL_14_LABELS)) - set(df.columns)
    if missing:
        raise SystemExit(f"[ERROR] {path} lacks columns {sorted(missing)}; is it CheXpert v1.0 {source}.csv?")
    df["label_source"] = f"chexpert_v1_{source}"
    return df


def resolve_paths(df: pd.DataFrame, pidx: dict) -> pd.DataFrame:
    norm = df["Path"].astype(str).str.replace("\\", "/", regex=False)
    parts = norm.str.extract(PATH_PATTERN, flags=re.IGNORECASE)
    parts.columns = ["sub", "patient", "study", "view"]
    file_paths = []
    for sub, pat, stu, view in zip(parts["sub"], parts["patient"], parts["study"], parts["view"]):
        found = None
        if isinstance(sub, str):
            sub = sub.lower()
            for root in pidx.get((sub, pat), []):
                cand = root / sub / pat / stu / f"{view}.png"
                if cand.exists():
                    found = str(cand)
                    break
        file_paths.append(found)
    out = df.copy()
    out["file_path"] = file_paths
    out["patient_id"] = parts["patient"]
    out["study_id"] = parts["patient"] + "/" + parts["study"]
    out["view_name"] = parts["view"]
    out["_parsed"] = parts["sub"].notna()
    return out


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    lab = df[FULL_14_LABELS].apply(pd.to_numeric, errors="coerce")
    bad = lab.notna() & ~lab.isin([1.0, 0.0, -1.0])
    if bad.any().any():
        raise SystemExit(f"[ERROR] Unexpected label values: {np.unique(lab.values[bad.values])[:5]}")
    df[FULL_14_LABELS] = lab.fillna(0).astype(np.int8)  # blank = not mentioned = negative
    return df


def stratified_patient_split(df: pd.DataFrame, tune_frac: float, test_frac: float, seed: int) -> dict:
    """Patient-level split stratified on the 5 competition pathologies (P/U/N per patient)."""
    agg = {}
    for lbl in COMPETITION_PATHOLOGIES:
        g = df.groupby("patient_id")[lbl]
        agg[lbl] = np.where(g.max() == 1, "P", np.where(g.min() == -1, "U", "N"))
    pats = pd.DataFrame(agg, index=df.groupby("patient_id").size().index)
    pats["strata"] = pats[COMPETITION_PATHOLOGIES].agg("".join, axis=1)
    pats = pats.sample(frac=1.0, random_state=seed)

    assign = {}
    for _, group in pats.groupby("strata"):
        ids = group.index.tolist()
        n = len(ids)
        n_tune = int(round(n * tune_frac))
        n_test = int(round(n * test_frac))
        if n < 3:
            n_tune = n_test = 0
        for pid in ids[:n_tune]:
            assign[pid] = "tune"
        for pid in ids[n_tune:n_tune + n_test]:
            assign[pid] = "test"
        for pid in ids[n_tune + n_test:]:
            assign[pid] = "train"
    return assign


def study_consistency(df: pd.DataFrame) -> tuple:
    """Fraction of multi-image studies whose images share an identical label vector."""
    key = df[FULL_14_LABELS].astype(str).agg("".join, axis=1)
    g = pd.DataFrame({"study": df["study_id"], "key": key}).groupby("study")["key"]
    sizes = g.size()
    multi = sizes[sizes > 1].index
    if len(multi) == 0:
        return float("nan"), 0
    nunique = g.nunique().loc[multi]
    return float((nunique == 1).mean()), int(len(multi))


def main():
    args = parse_args()
    t0 = time.time()
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)

    print("=" * 72)
    print(" Building split manifest from ORIGINAL CheXpert v1.0 labels")
    print("=" * 72)

    train = read_label_csv(Path(args.train_csv), "train")
    valid = read_label_csv(Path(args.valid_csv), "valid")
    print(f"[1/6] Label rows: train.csv={len(train):,} | valid.csv={len(valid):,}")

    print("[2/6] Indexing patient folders across PNG chunks...")
    pidx = index_patients(dataset_dir)
    print(f"      {len(pidx):,} patient folders indexed")

    print("[3/6] Matching label rows to PNG files on disk (this reads directory metadata; ~1-3 min on HDD)...")
    df = pd.concat([resolve_paths(train, pidx), resolve_paths(valid, pidx)], ignore_index=True)
    unparsed = int((~df["_parsed"]).sum())
    missing = int(df["file_path"].isna().sum())
    matched = len(df) - missing
    print(f"      matched {matched:,}/{len(df):,} rows ({matched / len(df):.2%}) | unparsable paths: {unparsed} "
          f"| no PNG on disk: {missing - unparsed}")
    for src in ("chexpert_v1_train", "chexpert_v1_valid"):
        sub = df[df["label_source"] == src]
        print(f"        {src}: {sub['file_path'].notna().sum():,}/{len(sub):,} matched")
    if matched / len(df) < 0.95:
        raise SystemExit("[ERROR] <95% of label rows found on disk. Check --dataset_dir and the CSV paths.")
    df = df[df["file_path"].notna()].reset_index(drop=True)

    df = clean_labels(df)
    df["is_frontal"] = df["view_name"].str.lower().str.contains("frontal")
    if "Frontal/Lateral" in df.columns:
        fl = df["Frontal/Lateral"].astype(str).str.strip().str.lower() == "frontal"
        disagree = int((fl != df["is_frontal"]).sum())
        if disagree:
            print(f"      note: {disagree} rows where 'Frontal/Lateral' disagrees with the file name (file name used)")
    df["view"] = np.where(df["is_frontal"], "frontal", "lateral")
    df["sex"] = df.get("Sex")
    df["age"] = df.get("Age")
    df["ap_pa"] = df.get("AP/PA")

    print("[4/6] Patient-level stratified split of train.csv patients + official valid split...")
    is_valid = df["label_source"] == "chexpert_v1_valid"
    assign = stratified_patient_split(df[~is_valid], args.tune_frac, args.test_frac, args.seed)
    df["split"] = np.where(is_valid, "valid", df["patient_id"].map(assign).fillna("train"))

    print("[5/6] Integrity checks...")
    pats = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "tune", "test", "valid")}
    names = list(pats)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = pats[names[i]] & pats[names[j]]
            if overlap:
                raise SystemExit(f"[ERROR] Patient leakage between {names[i]} and {names[j]}: {len(overlap)} patients")
    print("      PASS: zero patient overlap across train/tune/test/valid")
    cons, n_multi = study_consistency(df)
    print(f"      Study-level label consistency: {cons:.1%} of {n_multi:,} multi-image studies "
          f"(the CheXbert JSON manifest scored 9%)")
    if not (cons >= 0.95):
        raise SystemExit("[ERROR] Labels are not consistent within studies; refusing to write the manifest.")

    fr = df[df["is_frontal"]]
    print("\n      Frontal images per split:")
    for s in ("train", "tune", "test", "valid"):
        print(f"        {s:<6s}: {int((fr['split'] == s).sum()):>7,d}")
    print("\n      Competition pathology prevalence, frontal (pos% / uncertain%):")
    print(f"        {'Pathology':<18s}" + "".join(f"{s:>16s}" for s in ("train", "tune", "test", "valid")))
    for lbl in COMPETITION_PATHOLOGIES:
        cells = []
        for s in ("train", "tune", "test", "valid"):
            v = fr.loc[fr["split"] == s, lbl]
            cells.append(f"{(v == 1).mean() * 100:6.1f}/{(v == -1).mean() * 100:5.1f}" if len(v) else "-")
        print(f"        {lbl:<18s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n[6/6] Writing manifest...")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / MANIFEST_PARQUET.name
    out_csv = out_dir / MANIFEST_CSV.name
    for path in (out_parquet, out_csv):
        backup = path.with_name(path.stem + ".chexbert_misaligned" + path.suffix)
        if path.exists() and not backup.exists():
            shutil.move(str(path), str(backup))
            print(f"      previous manifest kept as {backup.name}")

    cols = ["image_id", "file_path", "patient_id", "study_id", "view", "is_frontal", *FULL_14_LABELS,
            "sex", "age", "ap_pa", "label_source", "split"]
    df["image_id"] = np.arange(len(df))
    df = df[cols]
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"      wrote {out_parquet} ({len(df):,} rows)")
    print(f"      wrote {out_csv}")
    print(f"\nDone in {time.time() - t0:.0f}s. Next: python tools\\label_sanity_probe.py  (expects mean AUC >= 0.57)")


if __name__ == "__main__":
    main()

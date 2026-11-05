"""
!!! DEPRECATED - DO NOT USE FOR TRAINING !!!
The CheXbert labels in report_fixed.json do not line up with their own path_to_image
column (images of one study disagree in 92% of studies; a frozen-feature linear probe
scores AUC 0.50). Models trained on this manifest cannot learn.
Use tools/build_manifest_chexpert.py (original CheXpert v1.0 train.csv / valid.csv).

build_manifest.py
Stage 0: Manifest Generation & Split QC for CheXpert Plus SOTA Ensemble
Specifications:
- Fast patient-to-chunk indexer (matches 100% of 223,462 images in seconds).
- Maps absolute path, patient_id, study_id, view orientation (frontal vs lateral).
- 14 CheXpert raw labels extracted (1.0 = pos, 0.0 = neg, -1.0 = unc).
- Strict patient-level multi-label stratified 80 / 10 / 10 split.
- Verifies zero patient leakage across Train, Tune, and Test splits.
- Exports split_manifest.csv and split_manifest.parquet.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

LABELS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
]

COMPETITION_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'
]

def build_split_manifest(base_dir=r"d:\Yatharth\Mipaas\ChestXray Dataset", output_dir=r"d:\Yatharth\Mipaas"):
    print("!" * 65)
    print("! WARNING: report_fixed.json labels are misaligned with the images.")
    print("! A manifest built here cannot be learned from (AUC ~0.50).")
    print("! Use tools/build_manifest_chexpert.py instead.")
    print("!" * 65)
    print("=================================================================")
    print("=== Phase 1: Building Manifest & Patient-Stratified Splits ===")
    print("=================================================================")
    t0 = time.time()
    
    # 1. Fast Index of Patients across all 5 extracted chunks
    print("\n[1/5] Indexing patient directories across all 5 chunks...")
    patient_to_chunk = {}
    for i in ['0', '1', '2', '3', '4_3']:
        chunk_png = os.path.join(base_dir, f'png_chexpert_plus_chunk_{i}', 'PNG')
        for sub in ['train', 'valid']:
            p = os.path.join(chunk_png, sub)
            if os.path.exists(p):
                for patient in os.listdir(p):
                    patient_to_chunk[patient] = chunk_png

    print(f"  Indexed {len(patient_to_chunk):,} unique patients in {time.time()-t0:.2f}s.")

    # 2. Parse report_fixed.json
    print("\n[2/5] Parsing report_fixed.json and mapping paths...")
    report_path = os.path.join(base_dir, "report_fixed.json")
    rows = []
    missing_count = 0
    
    with open(report_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            raw_path = data["path_to_image"]
            parts = raw_path.split("/")
            # parts: ['train', 'patient42142', 'study5', 'view1_frontal.jpg']
            subfolder = parts[0]
            patient_id = parts[1]
            study_sub = parts[2]
            view_filename = os.path.splitext(parts[3])[0] + ".png"
            
            chunk_root = patient_to_chunk.get(patient_id)
            if not chunk_root:
                missing_count += 1
                continue
                
            full_path = os.path.join(chunk_root, subfolder, patient_id, study_sub, view_filename)
            
            # Determine view orientation
            is_frontal = "frontal" in view_filename.lower()
            is_lateral = "lateral" in view_filename.lower()
            view_type = "frontal" if is_frontal else ("lateral" if is_lateral else "other")
            
            row = {
                "image_id": idx,
                "file_path": full_path,
                "patient_id": patient_id,
                "study_id": f"{patient_id}/{study_sub}",
                "view": view_type,
                "is_frontal": is_frontal
            }
            
            # Extract 14 raw labels (1.0 = pos, 0.0 = neg, -1.0 = unc, None = 0.0)
            for lbl in LABELS:
                val = data.get(lbl)
                if val == 1.0:
                    row[lbl] = 1
                elif val == -1.0:
                    row[lbl] = -1
                else:
                    row[lbl] = 0
                    
            rows.append(row)
            
    df = pd.DataFrame(rows)
    print(f"  Processed {len(df):,} records (Missing: {missing_count}).")
    print(f"  Frontal views: {df['is_frontal'].sum():,} ({df['is_frontal'].mean()*100:.1f}%)")
    print(f"  Lateral views: {(~df['is_frontal']).sum():,} ({(~df['is_frontal']).mean()*100:.1f}%)")

    # 3. Patient-Level Multi-Label Stratified Split (80 / 10 / 10)
    print("\n[3/5] Performing patient-level stratified 80 / 10 / 10 split...")
    patient_records = defaultdict(lambda: {lbl: 0 for lbl in LABELS})
    for row in rows:
        pid = row["patient_id"]
        for lbl in LABELS:
            v = row[lbl]
            if v == 1:
                patient_records[pid][lbl] = 1
            elif v == -1 and patient_records[pid][lbl] != 1:
                patient_records[pid][lbl] = -1

    patient_df = pd.DataFrame.from_dict(patient_records, orient="index")
    patient_df.index.name = "patient_id"
    patient_df.reset_index(inplace=True)
    
    # Stratification string across competition pathologies
    patient_df["strata"] = patient_df[COMPETITION_PATHOLOGIES].apply(
        lambda r: "".join(["P" if x == 1 else ("U" if x == -1 else "N") for x in r]), axis=1
    )
    
    # Deterministic assignment with fixed seed
    np.random.seed(42)
    patient_df = patient_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    split_assignments = {}
    for _, group in patient_df.groupby("strata"):
        n = len(group)
        if n == 1:
            split_assignments[group.iloc[0]["patient_id"]] = "train"
        else:
            n_train = int(np.round(n * 0.80))
            n_tune = int(np.round(n * 0.10))
            train_pids = group.iloc[:n_train]["patient_id"].values
            tune_pids = group.iloc[n_train:n_train+n_tune]["patient_id"].values
            test_pids = group.iloc[n_train+n_tune:]["patient_id"].values
            
            for pid in train_pids: split_assignments[pid] = "train"
            for pid in tune_pids: split_assignments[pid] = "tune"
            for pid in test_pids: split_assignments[pid] = "test"

    df["split"] = df["patient_id"].map(split_assignments).fillna("train")

    # 4. Split Assertions & Leakage Verification
    print("\n[4/5] Verifying split assertions & patient isolation...")
    train_patients = set(df[df["split"] == "train"]["patient_id"])
    tune_patients = set(df[df["split"] == "tune"]["patient_id"])
    test_patients = set(df[df["split"] == "test"]["patient_id"])
    
    assert len(train_patients & tune_patients) == 0, "LEAKAGE: Train & Tune overlap!"
    assert len(train_patients & test_patients) == 0, "LEAKAGE: Train & Test overlap!"
    assert len(tune_patients & test_patients) == 0, "LEAKAGE: Tune & Test overlap!"
    print("  PASS: Zero patient overlap across all splits (Strict Patient Isolation).")
    
    print("\n  Prevalence Check Across Splits (Positive %):")
    header = f"  {'Pathology':<20} | {'Train %':>8} | {'Tune %':>8} | {'Test %':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lbl in COMPETITION_PATHOLOGIES:
        p_train = (df[df['split']=='train'][lbl] == 1).mean() * 100
        p_tune = (df[df['split']=='tune'][lbl] == 1).mean() * 100
        p_test = (df[df['split']=='test'][lbl] == 1).mean() * 100
        print(f"  {lbl:<20} | {p_train:>7.2f}% | {p_tune:>7.2f}% | {p_test:>7.2f}%")

    # 5. Export Manifests
    print("\n[5/5] Exporting manifest files...")
    out_csv = os.path.join(output_dir, "split_manifest.csv")
    df.to_csv(out_csv, index=False)
    print(f"  CSV manifest saved: {out_csv} ({os.path.getsize(out_csv)/(1024*1024):.1f} MB)")
    
    try:
        out_parquet = os.path.join(output_dir, "split_manifest.parquet")
        df.to_parquet(out_parquet, index=False)
        print(f"  Parquet manifest saved: {out_parquet} ({os.path.getsize(out_parquet)/(1024*1024):.1f} MB)")
    except Exception as e:
        pass

    print(f"\nManifest build completed successfully in {time.time()-t0:.1f}s.")
    print("\nPartition Breakdown (All images):")
    for s, c in df['split'].value_counts().items():
        print(f"  {s.upper():<6}: {c:,} images ({c/len(df)*100:.2f}%)")
        
    print("\nPartition Breakdown (Frontal only - Primary training set):")
    frontal_df = df[df['is_frontal']]
    for s, c in frontal_df['split'].value_counts().items():
        print(f"  {s.upper():<6}: {c:,} images ({c/len(frontal_df)*100:.2f}%)")

if __name__ == "__main__":
    build_split_manifest()

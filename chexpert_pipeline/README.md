# CheXpert Plus Classification Pipeline

Multi-label chest X-ray classification on CheXpert Plus (153k frontal training images, 7 target
classes including the 5 CheXpert competition pathologies: Atelectasis, Cardiomegaly,
Consolidation, Edema, Pleural Effusion).

---

## 1. Why the previous models scored AUC 0.50 (root cause)

Both earlier runs, DenseNet-121 and the 111M-parameter TriBackboneFusion, stayed at
**AUC ≈ 0.50 on every class** for 2–3 epochs, with a flat training loss (0.624 → 0.616).
That is chance level. The cause is the data, not the model or the hyperparameters:

**The labels in `ChestXray Dataset/report_fixed.json` (CheXbert) are attached to the wrong images.**

| Check | Expected with correct labels | Measured on the old manifest |
|---|---|---|
| Frontal and lateral image of the same study (one report) carry identical labels | ~100% of studies | **7.6%** (9.1% after blank→0) |
| Logistic regression on frozen ImageNet DenseNet-121 features, 6k train / 2k tune images | ~0.64 mean (measured after the fix) | **0.49–0.53 on all 7 classes** |
| Fully trained models | ~0.80+ on held-out data | 0.50 |

The images themselves are fine: valid 8-bit chest X-rays, all files present (4 of 223,462 missing).
Shifting, sorting or re-ordering the label rows does not recover the alignment, so the labels must
come from another source. A related upstream report:
[Stanford-AIMI/chexpert-plus#13](https://github.com/Stanford-AIMI/chexpert-plus/issues/13)
(identical report text, opposite CheXbert labels).

**Fix:** use the original CheXpert v1.0 labels (`train.csv`, `valid.csv`). They are keyed by the
same `patient/study/view` paths as the CheXpert Plus PNGs. The `valid` folder on disk holds exactly
the 234 images (202 frontal) of the official, radiologist-labelled CheXpert validation set.

Two further problems made each epoch take ~9 hours:

* **Disk:** the 3–9 MB full-resolution PNGs sit on a SATA hard drive (D:). Reading them with
  `num_workers=0` ran at ~5 img/s; even 8 workers reach only ~20–28 img/s.
* **VRAM:** TriBackboneFusion at batch 12 / 384 px needs more than 12 GB. Windows drivers do not
  raise an error in that case; they quietly page GPU memory into system RAM ("sysmem fallback"),
  which multiplies step time.

---

## 2. Quick start

```
D:\Yatharth\Mipaas\
├── labels\                 <- put CheXpert v1.0 train.csv + valid.csv here
├── run_prepare_data.bat    <- 1) labels -> QC -> probe -> image cache
├── run_training.bat        <- 2) run_training.bat densenet121 | tf_efficientnetv2_s | rad_dino (| convnext_small)
├── run_ensemble.bat        <- 3) average all trained models
└── run_evaluate.bat        <- optional: flip-TTA + Youden thresholds for one checkpoint
```

**Step 0 (can start now; labels not needed):** build the image cache once. It takes ~1.5–2 h
because the hard drive is the bottleneck, writes ~13 GB to `C:\chexpert_cache\jpg512` (NVMe), and
can be resumed.
```
cd D:\Yatharth\Mipaas\chexpert_pipeline
C:\ProgramData\Anaconda3\python.exe tools\build_image_cache.py
```

**Step 1:** download CheXpert v1.0 `train.csv` + `valid.csv` (Stanford AIMI, same research-use
agreement) into `D:\Yatharth\Mipaas\labels\`.

**Step 2:** `run_prepare_data.bat`
1. `tools/build_manifest_chexpert.py`: new `split_manifest.parquet` with train/tune/test
   (patient-level, stratified, 90/5/5) plus the official `valid` split. The old manifest is kept
   as `split_manifest.chexbert_misaligned.*`. The script refuses to write if labels are not
   consistent within studies.
2. `qc_audit.py`: 6 gates, including **Gate 6 (label integrity)**, which blocks training on
   misaligned labels.
3. `tools/label_sanity_probe.py`: linear probe; it must PASS (mean AUC ≥ 0.57; real labels score ~0.64, misaligned ~0.50).
4. `tools/build_image_cache.py`: adds any images not cached yet.

**Step 3:** train two CNNs and one transformer (each run: QC → train → test + valid evaluation → prediction CSVs)
```
run_training.bat densenet121        (512 px, ~1.5 h)
run_training.bat tf_efficientnetv2_s (512 px, ~1.4 h)
run_training.bat rad_dino_frozen    (optional quick transformer baseline, ~25 min)
run_training.bat rad_dino           (518 px transformer, full fine-tune, ~3 h)
```
`convnext_small` (512 px, ~2 h) is available as a fourth ensemble member.

**Step 4:** `run_ensemble.bat` averages every trained model on the `valid` and `test` splits.

---

## 2b. Transformer: RAD-DINO

`rad_dino` is a Vision Transformer (ViT-B/14, 86.6M params) from Microsoft
([Pérez-García et al., *Nature Machine Intelligence* 2025](https://doi.org/10.1038/s42256-024-00965-w);
weights: [huggingface.co/microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino), MIT licence, research use).

* **Why this one:** it was pretrained (self-supervised, images only) on **882,775 chest X-rays**
  (MIMIC-CXR, CheXpert, NIH, PadChest, BRAX), while every other backbone here was pretrained on everyday photos.
  Its native input is **518 px**, which matches the 512 px plan, and it uses LayerNorm, so small batches don't hurt it.
* **How it is trained:** features = [CLS token ; mean of the 37×37 patch tokens] → LayerNorm → Linear(7).
  Full fine-tuning with **layer-wise learning-rate decay** (top block 5e-5, each lower block ×0.75, head 1e-3),
  weight decay 0.05, drop-path 0.1, batch 8 × 4 accumulation, 2 epochs, its own input normalization
  (mean 0.5307, std 0.2583).
* **Options:** `--freeze_blocks 12` trains only the head (`rad_dino_frozen` preset); `--freeze_blocks 8`
  fine-tunes only the last 4 blocks (~40 min/epoch).
* **Caveat for reports:** RAD-DINO's self-supervised pretraining included CheXpert images (also the
  validation images), though never their labels.
* It loads into `timm` (`vit_base_patch14_dinov2`) from `backbone_compatible.safetensors`; no
  `transformers` install is needed. The first run downloads 346 MB to the HuggingFace cache.
* Alternatives that were measured but not chosen: MaxViT-T (ImageNet, batch 8 only, 45 img/s) and
  Swin-V2-T (out of memory at 512 px).

---

## 3. What changed

| Area | Before | Now |
|---|---|---|
| Labels | CheXbert JSON, misaligned with the images | Original CheXpert v1.0 labels; Gate 6 + probe block bad labels |
| Final benchmark | random 10% of train patients | Official CheXpert **valid** set (radiologist consensus) + held-out test split |
| Image loading | full-res PNG from HDD, `num_workers=0` (~5 img/s) | 512 px JPEG cache on NVMe, 8 workers (GPU-bound) |
| Model | TriBackboneFusion, batch 12 (exceeds 12 GB) | One backbone per run (2 CNNs + RAD-DINO transformer), then an ensemble |
| Resolution | 384 px | 512 px CNNs, 518 px RAD-DINO (served from the 512 px cache) |
| Uncertain labels | U-Ones for every class | Per class (U-Ones: Atelectasis, Edema, Effusion, Opacity, Devices; U-Zeros: Cardiomegaly, Consolidation) + label smoothing (Pham et al. 2021) |
| Validation | once per epoch on 2,000 images | 3× per epoch on 5,000 images; best checkpoint kept; full splits at the end |
| VRAM | silent spill into system RAM | `--max_vram_fraction 0.92`: clear OOM error instead |
| evaluate.py | ignored the checkpoint's image size; silently used 2,000 tune images | uses checkpoint settings, full split, `--split valid`, `--tta` |
| ensemble.py | re-read labels from the manifest (row-order mismatch; uncertain counted as positive) | uses `true_*`/`unc_*` columns of the prediction CSVs; certain labels only |

Measured on this machine (RTX 4070 SUPER, AMP, GPU-only benchmark; epoch = ~172k training images):

| Model | Size | Batch | VRAM | Train speed | Epoch |
|---|---|---|---|---|---|
| densenet121 | 512 | 32 (16×2 used) | 10.5 GB | 114 img/s | ~25 min |
| tf_efficientnetv2_s | 512 | 16 | 6.3 GB | 120 img/s | ~24 min |
| convnext_small | 512 | 16 | 8.3 GB | 75 img/s | ~38 min |
| rad_dino, full fine-tune | 518 | 8 | 5.1 GB | 35 img/s | ~82 min |
| rad_dino, last 4 blocks | 518 | 32 | 6.2 GB | 72 img/s | ~40 min |
| rad_dino, frozen | 518 | 32 | 1.6 GB | 136 img/s | ~21 min |
| tri_nn_transformer | 384 | 12 | OOM (>11 GB) | – | – |
| tri_nn_transformer | 384 | 8 | 8.0 GB | 42 img/s | ~61 min |

At 384 px with real data loading from the cache, end-to-end speed was ~85% of the GPU-only figure
(densenet121 170 vs 200 img/s).

`channels_last` memory format made DenseNet and EfficientNetV2 ~6× slower here (27–40 img/s), so it
is not used.

---

## 4. Realistic expectations

* **Tune/test splits** (automatic labeler labels, noisy): competition mean AUC of roughly
  **0.80–0.85**.
* **Official valid set** (radiologist labels): roughly **0.87–0.90** for a single DenseNet-121 (the
  CheXpert paper's DenseNet-121 with U-Ones averaged ~0.89), with ensembles typically adding
  +0.01–0.02. The valid set has only 202 frontal images, so expect ±0.01–0.02 noise between runs.
* If the probe or Gate 6 fails, stop: no model setting can compensate for misaligned labels.

---

## 5. Files

```
chexpert_pipeline/
├── config.py               # paths (labels, cache), labels, uncertainty rules, default recipe
├── dataset.py              # LetterboxResize, cache-aware dataset, loaders (incl. create_eval_loader)
├── models.py               # single timm backbones + TriBackboneFusion
├── losses.py               # masked BCE (soft targets OK), ASL
├── metrics.py              # CheXpert AUC (certain labels), Youden thresholds, prediction export
├── train.py                # training engine (mid-epoch validation, VRAM cap, final test+valid eval)
├── evaluate.py             # checkpoint evaluation (--split valid|test|tune, --tta)
├── ensemble.py             # mean / weighted / rank ensembling of prediction CSVs
├── qc_audit.py             # 6-gate pre-flight audit (Gate 6 = label integrity)
├── verify_pipeline.py      # 8 plumbing tests (shapes, gradients, bias) - NOT a label check
└── tools/
    ├── build_manifest_chexpert.py   # manifest from CheXpert v1.0 train.csv/valid.csv
    ├── build_image_cache.py         # one-time 512 px JPEG cache
    ├── label_sanity_probe.py        # linear probe: do labels describe the images?
    └── build_manifest.py            # DEPRECATED (CheXbert JSON, misaligned)
```

The pre-fix code is preserved in `D:\Yatharth\Mipaas\chexpert_pipeline_backup_2026-11-03\`.

# Brain Tumor Segmentation

This repository contains scripts and example training code to experiment with brain tumor segmentation on two datasets:

- Medical Segmentation Decathlon Task01_BrainTumour (2.5D slice-based training)
- BraTS (3D full-volume / patch UNet training)

The code is meant for experiments and debugging (CPU or GPU). It includes dataset helpers, simple training loops, and evaluation tools.

---

## Prerequisites

- Python 3.8+ (venv recommended)
- If you have an NVIDIA GPU, install a CUDA-enabled PyTorch build to speed training.

Install dependencies inside a virtual environment (PowerShell example):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Notes for Windows:
- Use `num_workers=0` for DataLoader in the provided scripts to avoid multiprocessing issues on Windows. The scripts already set `num_workers=0` where appropriate.

---

## Repository layout (important files)

- `train.py` — 2.5D slice-based training (Task01 / Decathlon)
- `train_brats.py` — 3D UNet training for BraTS dataset
- `dataset_25d.py` — 2.5D dataset helpers (`TwoPointFiveDDataset`, `SliceDataset`)
- `dataset_brat20.py` — BraTS dataset loader (accepts `.nii` and `.nii.gz`, crops/pads to patch size)
- `model_unet3d.py` — simple 3D UNet used for BraTS experiments
- `model_train.py` — model factory, transforms and 2D training utilities
- `losses.py` — 3D dice loss helper
- `tools/` — helpers: `pred_stats.py`, `inspect_labels.py`, `eval_brats.py` (evaluator)
- `checkpoints/` — saved checkpoints and the `STOP` flag

---

## STOP file (safe stop)

Creating a file named `checkpoints/STOP` will make training loops stop gracefully (the code checks for this file each epoch). Remove it to resume training:

```powershell
Remove-Item .\checkpoints\STOP -ErrorAction SilentlyContinue
```

---

## Decathlon Task01_BrainTumour (2.5D)

1) Download / prepare dataset using MONAI (script included):

```powershell
python download_decathlon.py
```

This downloads and prepares Task01; the script sets `cache_rate=0.0` and `runtime_cache=True` to avoid loading everything in RAM.

2) Train (slice-based 2.5D):

```powershell
# Optionally set number of epochs for the run
# $env:TRAIN_EPOCHS=10
.\venv\Scripts\python.exe .\train.py
```

Notes:
- `train.py` uses `SliceDataset` for deterministic per-slice indexing and a `WeightedRandomSampler` to upweight tumor-containing slices.
- On Windows use `num_workers=0` in DataLoader (already set in scripts).
- Evaluate with `eval.py` (per-class dice) or `tools/pred_stats.py` (prediction voxel counts and slice-level stats).

---

## BraTS (3D)

1) Dataset layout expected by `dataset_brat20.py`:

```
data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
  BraTS20_Training_001/
    BraTS20_Training_001_flair.nii (or .nii.gz)
    BraTS20_Training_001_t1.nii
    BraTS20_Training_001_t1ce.nii
    BraTS20_Training_001_t2.nii
    BraTS20_Training_001_seg.nii
  BraTS20_Training_002/
  ...
```

Place the downloaded BraTS folders under `data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/`.

2) Run a smoke test (example):

```powershell
# $env:TRAIN_EPOCHS=1
# $env:BRATS_BATCH=1
.\venv\Scripts\python.exe .\train_brats.py
```

`train_brats.py` supports these env vars:
- `TRAIN_EPOCHS` — number of epochs
- `BRATS_BATCH` — batch size
- `LR` — learning rate

Notes:
- `dataset_brat20.py` will accept `.nii` or `.nii.gz` files and will center-crop/pad volumes to `patch_size` (default 128) so shapes match the UNet.
- Training on CPU is slow — a single epoch on CPU can take hours. Use a GPU for practical training.

---

## Evaluation

- Task01 (2D): run `eval.py` or `tools/pred_stats.py` for per-class Dice/prediction stats.
- BraTS (3D): use `tools/eval_brats.py`. For quick checks on CPU you can evaluate a small subset by setting `EVAL_SUBSET`:

```powershell
$env:EVAL_SUBSET=20
.\venv\Scripts\python.exe .\tools\eval_brats.py
```

---

## Debugging utilities

- `tools/inspect_labels.py` — list unique labels in validation set
- `tools/pred_stats.py` — predicted voxel counts per class over validation slices
- `tools/eval_brats.py` — per-class Dice evaluator (supports subset mode)

---

## Suggestions to improve tumor-class performance

1. Fix MRI normalization — MR images aren't 0–255. Use per-volume normalization (z-score or min-max per volume) in `dataset_brat20.get_transforms()`.
2. Use combined Dice + weighted Cross-Entropy (or Focal + Dice) with class weights computed from inverse frequency.
3. Oversample tumor-containing patches or use tumor-centered patch sampling so batches contain more tumor voxels.
4. Start with a small subset and a smaller UNet (`base_c` reduction) to iterate quickly. Move to full data + GPU once settings are stable.

---

## Quick PowerShell snippets

Activate the virtualenv:

```powershell
.\venv\Scripts\Activate.ps1
```

Run training with one-liners:

```powershell
# $env:TRAIN_EPOCHS=5; $env:BRATS_BATCH=1; .\venv\Scripts\python.exe .\train_brats.py
```

Remove the STOP flag:

```powershell
Remove-Item .\checkpoints\STOP -ErrorAction SilentlyContinue
```

---

If you'd like, I can:

- Patch `dataset_brat20` to use per-volume normalization and re-evaluate on a small subset.
- Add `--subset`, `--patch_size`, and `--base_c` CLI support to `train_brats.py` and run a short subset training with class-weighted loss.

Tell me which and I will implement and run it for you.

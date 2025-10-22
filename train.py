# from model_train import make_model, make_transforms, train_loop
# from dataset_25d import TwoPointFiveDDataset, load_nifti
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from sklearn.model_selection import train_test_split
# import glob, torch, os

# def main():
#     data_root = "data/Task01_BrainTumour/Task01_BrainTumour"
#     img_files = sorted(glob.glob(f"{data_root}/imagesTr/*.nii.gz"))
#     lbl_files = sorted(glob.glob(f"{data_root}/labelsTr/*.nii.gz"))

#     print("Found", len(img_files), "images and", len(lbl_files), "labels.")
#     if len(img_files) == 0:
#         raise ValueError("No training images found! Check folder paths or extensions.")

#     train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
#         img_files, lbl_files, test_size=0.2, random_state=42
#     )

#     train_tf, val_tf = make_transforms((256, 256))
#     # Use slice-level deterministic dataset so sampling/weights are per-slice
#     from dataset_25d import SliceDataset
#     train_ds = SliceDataset(train_imgs, train_lbls, transform=train_tf)
#     val_ds = SliceDataset(val_imgs, val_lbls, transform=val_tf)

#     # Sample Test
#     # quick test
#     x, y = train_ds[0]
#     print("Sample shape:", x.shape, y.shape, "dtype:", x.dtype)


#     # 👇 Set num_workers=0 on Windows to avoid multiprocessing issues
#     # Build a simple per-slice boolean indicating whether a slice contains any tumor voxels.
#     # This is used to construct a WeightedRandomSampler that upweights tumor-containing slices.
#     # Build deterministic per-slice tumor flags and class counts
#     import numpy as np
#     slice_has_tumor = []
#     class_counts = np.zeros(4, dtype=np.int64)
#     for i in range(len(train_ds)):
#         _, lbl = train_ds[i]
#         has = 1 if lbl.sum().item() > 0 else 0
#         slice_has_tumor.append(has)
#         vals, counts = np.unique(lbl.numpy().astype(np.int64), return_counts=True)
#         for v, c in zip(vals, counts):
#             if 0 <= v < 4:
#                 class_counts[v] += int(c)

#     # Set a higher weight for tumor-containing slices. Tune the factor if needed.
#     bg_weight = 1.0
#     tumor_weight = 4.0
#     sample_weights = [tumor_weight if s == 1 else bg_weight for s in slice_has_tumor]
#     sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

#     # Use the sampler for training; keep deterministic order for validation
#     train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler, num_workers=0)
#     val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # Use 4 input channels (FLAIR, T1w, t1gd, T2w) and 4 classes (background + 3 tumor labels)
#     # Switch to a smaller backbone for CPU-friendly training (efficientnet-b0)
#     model = make_model("efficientnet-b0", encoder_weights="imagenet", in_channels=4, classes=4, device=device)

#     # Reduced batch size for CPU (safer on RAM). Update DataLoader accordingly.
#     train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
#     val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

#     # Set epochs to a small number suitable for a ~1 hour run on CPU.
#     # Allow a quick smoke run by setting environment variable TRAIN_EPOCHS
#     try:
#         import os
#         epochs = int(os.environ.get('TRAIN_EPOCHS', '100'))
#     except Exception:
#         epochs = 100

#     print('Training voxel counts per class:', class_counts.tolist())
#     # Sqrt inverse-frequency weighting with clipping to avoid extreme weights
#     total = float(class_counts.sum())
#     inv_freq = total / (class_counts + 1e-8)
#     sqrt_inv = np.sqrt(inv_freq)
#     # clip weights to a reasonable range
#     clipped = np.clip(sqrt_inv, a_min=0.1, a_max=50.0)
#     class_weights = clipped.astype(float)
#     print('Using class weights (sqrt-inv clipped):', class_weights.tolist())

#     history = train_loop(
#         model, train_loader, val_loader,
#         epochs=epochs, lr=5e-4, save_dir="checkpoints", device=device, class_weights=class_weights
#     )
#     print("✅ Training completed successfully!")

# if __name__ == "__main__":
#     main()


# # --------------------------------------------------------------------------

# # BraTS Dataset Implementation

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from model_train import make_model, make_transforms, train_loop
from dataset_25d import SliceDataset


def main():
    data_root = "data/Task01_BrainTumour/Task01_BrainTumour"
    img_files = sorted(glob.glob(f"{data_root}/imagesTr/*.nii.gz"))
    lbl_files = sorted(glob.glob(f"{data_root}/labelsTr/*.nii.gz"))

    if len(img_files) == 0:
        raise ValueError(f"No images found in {data_root}. Put dataset there or run download script.")

    print("Found", len(img_files), "images and", len(lbl_files), "labels.")

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        img_files, lbl_files, test_size=0.2, random_state=42
    )

    train_tf, val_tf = make_transforms((256, 256))
    train_ds = SliceDataset(train_imgs, train_lbls, transform=train_tf)
    val_ds = SliceDataset(val_imgs, val_lbls, transform=val_tf)

    if len(train_ds) == 0:
        raise ValueError("SliceDataset has zero length. Check your label NIfTIs and shapes.")

    # Build deterministic per-slice tumor flags and class counts
    slice_has_tumor = []
    class_counts = np.zeros(4, dtype=np.int64)
    print("Scanning", len(train_ds), "slices to compute sampling weights and class frequencies...")
    for i in range(len(train_ds)):
        _, lbl = train_ds[i]
        has = 1 if lbl.sum().item() > 0 else 0
        slice_has_tumor.append(has)
        vals, counts = np.unique(lbl.numpy().astype(np.int64), return_counts=True)
        for v, c in zip(vals, counts):
            if 0 <= v < 4:
                class_counts[v] += int(c)

    print('Training voxel counts per class:', class_counts.tolist())

    # Sampling: upweight tumor slices deterministically
    bg_weight = 1.0
    tumor_weight = 4.0
    sample_weights = [tumor_weight if s == 1 else bg_weight for s in slice_has_tumor]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = make_model("efficientnet-b0", encoder_weights="imagenet", in_channels=4, classes=4, device=device)

    # compute class weights (sqrt-inv freq clipped)
    total = float(class_counts.sum())
    inv_freq = total / (class_counts + 1e-8)
    sqrt_inv = np.sqrt(inv_freq)
    clipped = np.clip(sqrt_inv, a_min=0.1, a_max=50.0)
    class_weights = clipped.astype(float)
    print('Using class weights (sqrt-inv clipped):', class_weights.tolist())

    # epochs via env var
    try:
        epochs = int(os.environ.get('TRAIN_EPOCHS', '20'))
    except Exception:
        epochs = 20

    # safety check: don't start if STOP exists
    stop_flag = os.path.join('checkpoints', 'STOP')
    if os.path.exists(stop_flag):
        print(f"Found {stop_flag} — training will not start until it's removed.")
        return

    train_loop(
        model, train_loader, val_loader,
        epochs=epochs, lr=5e-4, save_dir="checkpoints", device=device, class_weights=class_weights
    )


if __name__ == '__main__':
    main()

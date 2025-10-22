import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random

def load_nifti(path):
    try:
        img = nib.load(path)
        data = img.get_fdata()
        return data.astype(np.float32)
    except Exception as e:
        print(f"[❌ NIfTI Load Error] {path}: {e}")
        return None

class TwoPointFiveDDataset(Dataset):
    def __init__(self, img_files, lbl_files, transform=None):
        self.img_files = img_files
        self.lbl_files = lbl_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        lbl_path = self.lbl_files[idx]

        img = load_nifti(img_path)
        lbl = load_nifti(lbl_path)

        if img is None or lbl is None:
            print(f"[⚠️ Skipping {idx}: Corrupt file]")
            return self.__getitem__((idx + 1) % len(self))

        # Expect (240, 240, 155, 4) for image, (240, 240, 155) for label
        if img.ndim == 3:
            img = np.expand_dims(img, axis=-1)
        elif img.ndim != 4:
            print(f"[⚠️ Dataset Warning] Skipping index {idx}: Invalid NIfTI dimensions {img.shape}")
            return self.__getitem__((idx + 1) % len(self))

        if lbl.ndim != 3:
            print(f"[⚠️ Label Warning] Skipping index {idx}: Invalid label dimensions {lbl.shape}")
            return self.__getitem__((idx + 1) % len(self))

        # Normalize image
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

        # Select a random slice
        z = random.randint(0, img.shape[2] - 1)
        img_slice = img[:, :, z, :]
        lbl_slice = lbl[:, :, z]
        # Ensure numpy dtypes for albumentations
        img_slice = img_slice.astype(np.float32)
        lbl_slice = lbl_slice.astype(np.int64)

        # Apply transforms using numpy inputs (albumentations expects numpy arrays)
        if self.transform:
            augmented = self.transform(image=img_slice, mask=lbl_slice)
            img_a = augmented.get("image")
            lbl_a = augmented.get("mask")

            # albumentations' ToTensorV2 returns torch tensors; handle both cases
            if isinstance(img_a, np.ndarray):
                img_tensor = torch.from_numpy(img_a.transpose(2, 0, 1)).float()
            else:
                # assume torch Tensor
                img_tensor = img_a.float()

            if isinstance(lbl_a, np.ndarray):
                lbl_tensor = torch.from_numpy(lbl_a).long()
            else:
                lbl_tensor = lbl_a.long()
        else:
            # No transforms: convert to torch tensors (C, H, W)
            img_tensor = torch.from_numpy(img_slice.transpose(2, 0, 1)).float()
            lbl_tensor = torch.from_numpy(lbl_slice).long()

        return img_tensor, lbl_tensor


class SliceDataset(Dataset):
    """
    Deterministic slice-level dataset.
    Builds an index of (img_path, lbl_path, z) so sampling and weighting can be done per-slice.
    Returns (image_tensor(C,H,W), label_tensor(H,W)).
    """
    def __init__(self, img_files, lbl_files, transform=None):
        assert len(img_files) == len(lbl_files), "image and label lists must match"
        self.entries = []  # list of tuples (img_path, lbl_path, z)
        self.transform = transform
        for img_path, lbl_path in zip(img_files, lbl_files):
            try:
                lbl_img = nib.load(lbl_path)
                lbl_shape = lbl_img.shape
                # expect (H, W, Z)
                if len(lbl_shape) == 3:
                    nz = lbl_shape[2]
                elif len(lbl_shape) == 4:
                    # unusual but handle
                    nz = lbl_shape[2]
                else:
                    nz = lbl_shape[-1]
                for z in range(nz):
                    self.entries.append((img_path, lbl_path, z))
            except Exception as e:
                print(f"[❌ SliceDataset read error] {lbl_path}: {e}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, lbl_path, z = self.entries[idx]
        img = load_nifti(img_path)
        lbl = load_nifti(lbl_path)

        if img is None or lbl is None:
            raise RuntimeError(f"Failed to load: {img_path} or {lbl_path}")

        # Expect img shape (H,W,Z,4) or (H,W,Z)
        if img.ndim == 3:
            img = np.expand_dims(img, axis=-1)

        img_slice = img[:, :, z, :]
        lbl_slice = lbl[:, :, z]

        img_slice = img_slice.astype(np.float32)
        lbl_slice = lbl_slice.astype(np.int64)

        if self.transform:
            augmented = self.transform(image=img_slice, mask=lbl_slice)
            img_a = augmented.get('image')
            lbl_a = augmented.get('mask')
            if isinstance(img_a, np.ndarray):
                img_tensor = torch.from_numpy(img_a.transpose(2, 0, 1)).float()
            else:
                img_tensor = img_a.float()
            if isinstance(lbl_a, np.ndarray):
                lbl_tensor = torch.from_numpy(lbl_a).long()
            else:
                lbl_tensor = lbl_a.long()
        else:
            img_tensor = torch.from_numpy(img_slice.transpose(2, 0, 1)).float()
            lbl_tensor = torch.from_numpy(lbl_slice).long()

        return img_tensor, lbl_tensor

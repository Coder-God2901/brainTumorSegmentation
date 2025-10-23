# dataset_brat20.py
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, NormalizeIntensityd,
    ScaleIntensityRanged, ToTensord
)
from glob import glob

class BraTS2020Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, patch_size=(128,128,128)):
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = patch_size

        # all patient folders
        raw_patients = sorted(glob(os.path.join(root_dir, 'MICCAI_BraTS2020_TrainingData', 'BraTS20_Training_*')))

        # Filter out any patient folders that are missing expected modality or seg files
        valid_patients = []
        required_modalities = ['flair', 't1', 't1ce', 't2']
        exts = ['.nii.gz', '.nii']
        for p in raw_patients:
            base = os.path.basename(p)
            ok = True
            # check modalities (accept either .nii.gz or .nii)
            for m in required_modalities:
                found = False
                for e in exts:
                    expected = os.path.join(p, f"{base}_{m}{e}")
                    if os.path.exists(expected):
                        found = True
                        break
                if not found:
                    ok = False
                    break
            # check segmentation
            seg_found = False
            for e in exts:
                seg_expected = os.path.join(p, f"{base}_seg{e}")
                if os.path.exists(seg_expected):
                    seg_found = True
                    break
            if not seg_found:
                ok = False
            if ok:
                valid_patients.append(p)

        self.patients = valid_patients
        n = len(self.patients)
        # split into train/val
        if split == 'train':
            self.patients = self.patients[:int(0.8*n)]
        else:
            self.patients = self.patients[int(0.8*n):]

    def __len__(self):
        return len(self.patients)

    def load_patient(self, patient_path):
        modalities = ['flair', 't1', 't1ce', 't2']
        imgs = []
        for m in modalities:
            # Try .nii.gz first, then .nii
            base = os.path.basename(patient_path)
            img = None
            for ext in ('.nii.gz', '.nii'):
                pth = os.path.join(patient_path, f"{base}_{m}{ext}")
                if os.path.exists(pth):
                    # request float32 directly to avoid temporary float64 allocation
                    img = nib.load(pth).get_fdata(dtype=np.float32)
                    break
            if img is None:
                raise FileNotFoundError(f"Missing modality file for {base}_{m} in {patient_path}")
            imgs.append(img)
        img = np.stack(imgs, axis=0).astype(np.float32)
        # load segmentation (accept .nii.gz or .nii)
        seg = None
        for ext in ('.nii.gz', '.nii'):
            seg_p = os.path.join(patient_path, f"{os.path.basename(patient_path)}_seg{ext}")
            if os.path.exists(seg_p):
                # nibabel requires floating dtype for get_fdata; request float32 then cast to int64
                seg = nib.load(seg_p).get_fdata(dtype=np.float32).astype(np.int64)
                break
        if seg is None:
            raise FileNotFoundError(f"Missing segmentation file for {os.path.basename(patient_path)} in {patient_path}")
        # Remap segmentation labels to 0..(n_classes-1).
        # Common BraTS labels are {0,1,2,4} — map 4 -> 3 so we have classes 0,1,2,3.
        seg = seg.astype(np.int64)
        if np.any(seg == 4):
            seg[seg == 4] = 3
        # Ensure labels are non-negative and reasonably bounded
        seg[seg < 0] = 0
        return img, seg

    def __getitem__(self, idx):
        img, seg = self.load_patient(self.patients[idx])
        # Ensure volume has exact patch_size dimensions (center-crop if larger, symmetric pad if smaller)
        _, H, W, D = img.shape
        px, py, pz = self.patch_size

        def center_crop_pad_array(vol, target_shape):
            """Center-crop or symmetric-pad numpy array vol to target_shape.
            vol: numpy array with spatial dims at the end (e.g., (C,H,W,D) or (H,W,D)).
            target_shape: tuple of 3 ints (tx,ty,tz) for spatial dims.
            Returns array with spatial dims equal to target_shape.
            """
            vol = vol.copy()
            # determine number of leading (non-spatial) dims
            spatial_axes = vol.shape[-3:]
            # iterate over the 3 spatial axes
            for i, target in enumerate(target_shape):
                axis = vol.ndim - 3 + i
                size = vol.shape[axis]
                if size > target:
                    start = (size - target) // 2
                    # build slice tuple
                    sl = [slice(None)] * vol.ndim
                    sl[axis] = slice(start, start + target)
                    vol = vol[tuple(sl)]
                elif size < target:
                    pad_before = (target - size) // 2
                    pad_after = target - size - pad_before
                    pads = [(0,0)] * vol.ndim
                    pads[axis] = (pad_before, pad_after)
                    vol = np.pad(vol, pads, mode='constant')
            return vol

        # apply to image (C,H,W,D)
        img = center_crop_pad_array(img, (px, py, pz))

        # ensure seg is H,W,D
        if seg.ndim == 4 and seg.shape[0] == 1:
            seg = seg[0]
        elif seg.ndim == 4 and seg.shape[0] != 1:
            # if seg unexpectedly has channel dim >1, collapse first axis
            seg = seg[0]
        seg = center_crop_pad_array(seg, (px, py, pz))

        sample = {'image': img, 'mask': seg}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['mask']


def get_transforms():
    # Use per-volume normalization for MRI (z-score) instead of assuming 0-255 ranges
    return Compose([
        # NormalizeIntensityd will compute (x - mean) / std per volume (nonzero voxels only)
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=['image', 'mask'], prob=0.5, max_k=3),
        ToTensord(keys=['image', 'mask'])
    ])

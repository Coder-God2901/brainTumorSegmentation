import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import glob
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_brat20 import BraTS2020Dataset, get_transforms
from model_unet3d import UNet3D


def compute_per_class_dice(preds, targets, num_classes=4, eps=1e-6):
    # preds, targets: numpy arrays (N, H, W, D)
    dices = np.zeros(num_classes, dtype=float)
    counts = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        pred_c = (preds == c).astype(np.uint8)
        targ_c = (targets == c).astype(np.uint8)
        inter = (pred_c & targ_c).sum()
        union = pred_c.sum() + targ_c.sum()
        if union == 0:
            # no voxels for this class in both pred & target
            dices[c] = float('nan')
        else:
            dices[c] = (2.0 * inter + eps) / (union + eps)
    return dices


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = 'data/BraTS2020_TrainingData'
    val_ds = BraTS2020Dataset(root_dir=root, split='val', transform=get_transforms())
    # Optionally evaluate only a subset for speed (set EVAL_SUBSET env var)
    try:
        subset_n = int(os.environ.get('EVAL_SUBSET', '0'))
    except Exception:
        subset_n = 0
    if subset_n > 0:
        subset_n = min(subset_n, len(val_ds))
        val_ds = Subset(val_ds, list(range(subset_n)))
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNet3D(in_channels=4, n_classes=4)
    ckpt = os.path.join('checkpoints','brats','best_model_brats.pth')
    if not os.path.exists(ckpt):
        print('Checkpoint not found at', ckpt)
        return
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    all_preds = []
    all_targs = []
    with torch.no_grad():
        for img, mask in loader:
            # img: tensor C,H,W,D (monai ToTensord gives tensor)
            img = img.to(device).float()
            mask = mask.numpy()
            out = model(img)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            # pred shape: (1, H, W, D)
            all_preds.append(pred[0])
            all_targs.append(mask[0])

    preds = np.stack(all_preds, axis=0)
    targs = np.stack(all_targs, axis=0)
    dices = compute_per_class_dice(preds, targs, num_classes=4)
    for i, d in enumerate(dices):
        if np.isnan(d):
            print(f'Class {i} Dice: n/a (no voxels in pred & gt)')
        else:
            print(f'Class {i} Dice: {d:.4f}')
    valid = [d for d in dices if not np.isnan(d)]
    print('Mean Dice (over present classes):', np.mean(valid) if len(valid) else float('nan'))


if __name__ == '__main__':
    main()

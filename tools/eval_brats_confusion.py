import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_brat20 import BraTS2020Dataset, get_transforms
from model_unet3d import UNet3D


def per_class_dice_and_confusion(preds, targs, num_classes=4):
    # preds, targs: numpy arrays shaped (N, H, W, D)
    conf = np.zeros((num_classes, num_classes), dtype=int)  # rows: truth, cols: pred
    dices = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        pred_c = (preds == c).astype(np.uint8)
        targ_c = (targs == c).astype(np.uint8)
        inter = (pred_c & targ_c).sum()
        union = pred_c.sum() + targ_c.sum()
        if union == 0:
            dices[c] = float('nan')
        else:
            dices[c] = (2.0 * inter) / union
    for t in range(num_classes):
        for p in range(num_classes):
            conf[t, p] = int(((targs == t) & (preds == p)).sum())
    return dices, conf


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = 'data/BraTS2020_TrainingData'
    val_ds = BraTS2020Dataset(root_dir=root, split='val', transform=get_transforms())
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
            img = img.to(device).float()
            out = model(img)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.append(pred[0])
            all_targs.append(mask.numpy()[0])

    preds = np.stack(all_preds, axis=0)
    targs = np.stack(all_targs, axis=0)
    dices, conf = per_class_dice_and_confusion(preds, targs, num_classes=4)
    print('Per-class Dice:')
    for i, d in enumerate(dices):
        print(f'Class {i}:', 'n/a' if np.isnan(d) else f'{d:.4f}')
    print('\nConfusion matrix (rows=true, cols=pred)):')
    print(conf)
    valid = [d for d in dices if not np.isnan(d)]
    print('\nMean Dice (over present classes):', np.mean(valid) if len(valid) else float('nan'))

if __name__ == '__main__':
    main()

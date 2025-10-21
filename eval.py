import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_25d import TwoPointFiveDDataset
from model_train import make_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def compute_dice_per_class(pred, true, num_classes):
    dices = []
    for c in range(num_classes):
        p = (pred == c).astype(np.uint8)
        t = (true == c).astype(np.uint8)
        inter = (p & t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dices.append(float('nan'))
        else:
            dices.append((2.0 * inter) / (union + 1e-8))
    return dices


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = 'data/Task01_BrainTumour/Task01_BrainTumour'
    imgs = sorted(glob.glob(f"{data_root}/imagesTr/*.nii.gz"))
    lbls = sorted(glob.glob(f"{data_root}/labelsTr/*.nii.gz"))

    # same split as train.py
    _, val_imgs, _, val_lbls = train_test_split(imgs, lbls, test_size=0.2, random_state=42)

    ds = TwoPointFiveDDataset(val_imgs, val_lbls, transform=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    num_classes = 4
    model = make_model('efficientnet-b0', encoder_weights=None, in_channels=4, classes=num_classes, device=device)
    ckpt_path = 'checkpoints/best_model.pth'
    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        return

    model.eval()

    dices_sum = np.zeros(num_classes, dtype=float)
    counts = np.zeros(num_classes, dtype=int)

    with torch.no_grad():
        for img_t, lbl_t in tqdm(loader, total=len(loader)):
            # img_t shape: [C,H,W] because dataset returns non-batched tensors; ensure batch dim
            if img_t.ndim == 3:
                img = img_t.unsqueeze(0).to(device).float()
            else:
                img = img_t.to(device).float()

            logits = model(img)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            true = lbl_t.numpy()
            # if lbl_t is shape [1,H,W] or [H,W]
            if true.ndim == 3 and true.shape[0] == 1:
                true = true[0]

            dices = compute_dice_per_class(pred, true, num_classes)
            for i, d in enumerate(dices):
                if not np.isnan(d):
                    dices_sum[i] += d
                    counts[i] += 1

    mean_dices = dices_sum / (counts + 1e-8)
    for i, md in enumerate(mean_dices):
        print(f'Class {i} Dice: {md:.4f}')
    print('Mean Dice (over classes):', np.nanmean(mean_dices))


if __name__ == '__main__':
    main()
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_25d import TwoPointFiveDDataset
from model_train import make_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def compute_dice_per_class(pred, true, num_classes):
    dices = []
    for c in range(num_classes):
        p = (pred == c).astype(np.uint8)
        t = (true == c).astype(np.uint8)
        inter = (p & t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dices.append(float('nan'))
        else:
            dices.append((2.0 * inter) / (union + 1e-8))
    return dices


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = 'data/Task01_BrainTumour/Task01_BrainTumour'
    imgs = sorted(glob.glob(f"{data_root}/imagesTr/*.nii.gz"))
    lbls = sorted(glob.glob(f"{data_root}/labelsTr/*.nii.gz"))

    # same split as train.py
    _, val_imgs, _, val_lbls = train_test_split(imgs, lbls, test_size=0.2, random_state=42)

    ds = TwoPointFiveDDataset(val_imgs, val_lbls, transform=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    num_classes = 4
    # Use the same backbone that was used for training
    model = make_model('efficientnet-b0', encoder_weights=None, in_channels=4, classes=num_classes, device=device)
    ckpt_path = 'checkpoints/best_model.pth'
    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        return

    model.eval()

    dices_sum = np.zeros(num_classes, dtype=float)
    counts = np.zeros(num_classes, dtype=int)

    with torch.no_grad():
        for img_t, lbl_t in tqdm(loader, total=len(loader)):
            # img_t shape: [C,H,W] because dataset returns non-batched tensors; ensure batch dim
            if img_t.ndim == 3:
                img = img_t.unsqueeze(0).to(device).float()
            else:
                img = img_t.to(device).float()

            logits = model(img)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            true = lbl_t.numpy()
            # if lbl_t is shape [1,H,W] or [H,W]
            if true.ndim == 3 and true.shape[0] == 1:
                true = true[0]

            dices = compute_dice_per_class(pred, true, num_classes)
            for i, d in enumerate(dices):
                if not np.isnan(d):
                    dices_sum[i] += d
                    counts[i] += 1

    mean_dices = dices_sum / (counts + 1e-8)
    for i, md in enumerate(mean_dices):
        print(f'Class {i} Dice: {md:.4f}')
    print('Mean Dice (over classes):', np.nanmean(mean_dices))


if __name__ == '__main__':
    main()
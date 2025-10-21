import glob
import numpy as np
import torch
import os, sys
from torch.utils.data import DataLoader
# ensure project root is on sys.path so sibling modules resolve when running from tools/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_25d import TwoPointFiveDDataset
from model_train import make_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = 'data/Task01_BrainTumour/Task01_BrainTumour'
    imgs = sorted(glob.glob(f"{data_root}/imagesTr/*.nii.gz"))
    lbls = sorted(glob.glob(f"{data_root}/labelsTr/*.nii.gz"))
    _, val_imgs, _, val_lbls = train_test_split(imgs, lbls, test_size=0.2, random_state=42)

    ds = TwoPointFiveDDataset(val_imgs, val_lbls, transform=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    num_classes = 4
    model = make_model('efficientnet-b0', encoder_weights=None, in_channels=4, classes=num_classes, device=device)
    try:
        state = torch.load('checkpoints/best_model.pth', map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print('Error loading checkpoint:', e)
        return

    model.eval()

    class_counts = np.zeros(num_classes, dtype=int)
    slices_with_nonbg = 0

    with torch.no_grad():
        for img_t, _ in tqdm(loader, total=len(loader)):
            img = img_t.unsqueeze(0).to(device).float() if img_t.ndim==3 else img_t.to(device).float()
            if img.ndim == 3:
                img = img.unsqueeze(0)
            logits = model(img)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            uniques, counts = np.unique(pred, return_counts=True)
            for u, c in zip(uniques, counts):
                class_counts[int(u)] += int(c)
            if np.any(uniques != 0):
                slices_with_nonbg += 1

    total_voxels = class_counts.sum()
    print('Predicted class voxel counts:', class_counts.tolist())
    print('Predicted class voxel percentages:', (class_counts/ (total_voxels+1e-8)).tolist())
    print('Number of slices with any non-background prediction:', slices_with_nonbg, 'out of', len(loader))

if __name__ == '__main__':
    main()

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_brat20 import BraTS2020Dataset, get_transforms
from model_unet3d import UNet3D
from losses import dice_loss


def main():
    # Config
    # Try a few common locations where BraTS data might be placed and pick the first that exists
    candidates = [
        "data/BraTS2020",
        "data/BraTS2020_TrainingData",
        "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        "data",
    ]
    root = None
    for c in candidates:
        if os.path.exists(c):
            root = c
            break
    if root is None:
        raise RuntimeError("No BraTS data folder found in expected locations: " + ",".join(candidates))
    print(f"Using BraTS root: {root}")
    batch_size = int(os.environ.get('BRATS_BATCH', '1'))
    epochs = int(os.environ.get('TRAIN_EPOCHS', '10'))
    lr = float(os.environ.get('LR', '1e-4'))
    save_dir = os.path.join('checkpoints', 'brats')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Dataset
    train_ds = BraTS2020Dataset(root_dir=root, split='train', transform=get_transforms())
    val_ds = BraTS2020Dataset(root_dir=root, split='val', transform=get_transforms())

    if len(train_ds) == 0:
        raise RuntimeError(f'No BraTS training data found under {root}. Ensure dataset is downloaded and laid out as expected.')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet3D(in_channels=4, n_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stop_flag = os.path.join('checkpoints', 'STOP')

    for epoch in range(1, epochs + 1):
        if os.path.exists(stop_flag):
            print(f'Found stop flag {stop_flag}. Stopping before epoch {epoch}.')
            break

        model.train()
        train_loss = 0.0
        for img, mask in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = dice_loss(out, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        print(f'Epoch {epoch} | Train Loss: {train_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f'Epoch {epoch} [Val]'):
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                loss = dice_loss(out, mask)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        print(f'Epoch {epoch} | Val Loss: {val_loss:.4f}')

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_brats_epoch{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_brats.pth'))
    print('Training finished.')


if __name__ == '__main__':
    main()

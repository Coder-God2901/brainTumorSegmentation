import time
import glob
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from dataset_25d import TwoPointFiveDDataset
from model_train import make_model, make_transforms, train_loop

# Configure run (3 hours default)
TOTAL_MINUTES = 180
# Conservative per-epoch estimate in seconds (use prior runs ~5.5 minutes -> 330s)
EST_EPOCH_SECONDS = 330
BATCH_SIZE = 4
NUM_WORKERS = 0  # Windows safety
IMAGE_SIZE = (256, 256)


def main(total_minutes=TOTAL_MINUTES, est_epoch_seconds=EST_EPOCH_SECONDS):
    total_seconds = total_minutes * 60
    # estimate
    est_epochs = max(1, int(total_seconds / est_epoch_seconds))

    print(f"Planned time: {total_minutes} min ({total_seconds} s).\n"
          f"Estimated epoch time: {est_epoch_seconds} s.\n"
          f"Estimated epochs to run: {est_epochs}")

    # dataset
    data_root = 'data/Task01_BrainTumour/Task01_BrainTumour'
    img_files = sorted(glob.glob(f"{data_root}/imagesTr/*.nii.gz"))
    lbl_files = sorted(glob.glob(f"{data_root}/labelsTr/*.nii.gz"))
    if len(img_files) == 0:
        raise RuntimeError('No training images found.')

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(img_files, lbl_files, test_size=0.2, random_state=42)

    train_tf, val_tf = make_transforms(IMAGE_SIZE)
    train_ds = TwoPointFiveDDataset(train_imgs, train_lbls, transform=train_tf)
    val_ds = TwoPointFiveDDataset(val_imgs, val_lbls, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    model = make_model('efficientnet-b0', encoder_weights='imagenet', in_channels=4, classes=4, device=device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('checkpoints', f'timed_{timestamp}')

    # run training for estimated epochs
    start = time.time()
    train_loop(model, train_loader, val_loader, epochs=est_epochs, lr=5e-4, save_dir=save_dir, device=device)
    elapsed = time.time() - start
    print(f'Training finished. Elapsed time: {elapsed/60:.2f} minutes')


if __name__ == '__main__':
    main()

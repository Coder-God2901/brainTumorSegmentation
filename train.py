from model_train import make_model, make_transforms, train_loop
from dataset_25d import TwoPointFiveDDataset, load_nifti
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import glob, torch, os

def main():
    data_root = "data/Task01_BrainTumour/Task01_BrainTumour"
    img_files = sorted(glob.glob(f"{data_root}/imagesTr/*.nii.gz"))
    lbl_files = sorted(glob.glob(f"{data_root}/labelsTr/*.nii.gz"))

    print("Found", len(img_files), "images and", len(lbl_files), "labels.")
    if len(img_files) == 0:
        raise ValueError("No training images found! Check folder paths or extensions.")

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        img_files, lbl_files, test_size=0.2, random_state=42
    )

    train_tf, val_tf = make_transforms((256, 256))
    train_ds = TwoPointFiveDDataset(train_imgs, train_lbls, transform=train_tf)
    val_ds = TwoPointFiveDDataset(val_imgs, val_lbls, transform=val_tf)

    # Sample Test
    # quick test
    x, y = train_ds[0]
    print("Sample shape:", x.shape, y.shape, "dtype:", x.dtype)


    # 👇 Set num_workers=0 on Windows to avoid multiprocessing issues
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use 4 input channels (FLAIR, T1w, t1gd, T2w) and 4 classes (background + 3 tumor labels)
    # Switch to a smaller backbone for CPU-friendly training (efficientnet-b0)
    model = make_model("efficientnet-b0", encoder_weights="imagenet", in_channels=4, classes=4, device=device)

    # Reduced batch size for CPU (safer on RAM). Update DataLoader accordingly.
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Set epochs to a small number suitable for a ~1 hour run on CPU.
    history = train_loop(
        model, train_loader, val_loader,
        epochs=100, lr=5e-4, save_dir="checkpoints", device=device
    )
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# ---------- Model Definition ----------
def make_model(encoder_name="efficientnet-b4", encoder_weights="imagenet",
               in_channels=3, classes=3, device="cuda"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    return model.to(device)

# ---------- Data Augmentation ----------
def make_transforms(image_size=(256,256)):
    train_tf = A.Compose([
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
        A.Normalize(),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Resize(*image_size),
        A.Normalize(),
        ToTensorV2()
    ])
    return train_tf, val_tf

# ---------- Custom Loss Functions ----------
class FocalDiceLoss(nn.Module):
    """
    Dice + (optionally weighted) CrossEntropy loss.
    If class_weights is provided, it's applied to the CE term.
    """
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        # store weights as a tensor or None
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None)

    def forward(self, inputs, targets):
        # inputs: logits (N, C, H, W), targets: (N, H, W) long
        probs = torch.softmax(inputs, dim=1)
        # DiceLoss in segmentation_models_pytorch expects probabilities and integer labels for multiclass mode
        dice_loss = self.dice(probs, targets)

        # Use standard CrossEntropyLoss with optional per-class weights
        # CrossEntropyLoss expects raw logits
        if self.class_weights is not None:
            # ensure weights are on the same device as inputs
            weights = self.class_weights.to(inputs.device)
        else:
            weights = None

        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=weights)

        # Combine losses (scale CE to keep similar magnitude as before)
        return dice_loss + 0.5 * ce_loss

# ---------- Training Loop ----------
def train_loop(model, train_loader, val_loader, epochs=100, lr=5e-4, device="cuda", save_dir="checkpoints", class_weights=None):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Use automatic mixed precision only when training on CUDA devices.
    use_amp = (device == "cuda" or (device == "cpu" and torch.cuda.is_available())) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    loss_fn = FocalDiceLoss(class_weights=class_weights)

    for epoch in range(1, epochs+1):
        # Allow an external stop signal by touching a file named STOP inside the save_dir.
        # If present at the start of an epoch, we still run the epoch but will break after saving
        # so the model terminates cleanly at an epoch boundary.
        stop_flag = os.path.join(save_dir, 'STOP')
        if os.path.exists(stop_flag):
            print(f"Found stop flag at {stop_flag}. Will stop after current epoch {epoch-1} (no new epochs will start).")
            # We intentionally do not break immediately so the user observing the run can see the message
            # and allow the epoch to finish cleanly; break will happen after saving below.
            force_stop_next = True
        else:
            force_stop_next = False
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if use_amp:
                # CUDA mixed precision
                with autocast():
                    preds = model(xb)
                    loss = loss_fn(preds, yb.long())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Plain FP32 on CPU (or when AMP not available)
                preds = model(xb)
                loss = loss_fn(preds, yb.long())
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb.long())
                val_loss += loss.item()

        print(f"Epoch {epoch}: Train Loss={total_loss/len(train_loader):.4f} | Val Loss={val_loss/len(val_loader):.4f}")
        torch.save(model.state_dict(), f"{save_dir}/model_epoch{epoch}.pth")

        # If a stop flag file was created, stop after finishing this epoch so we terminate at an epoch boundary.
        if os.path.exists(stop_flag) or force_stop_next:
            print(f"Stop flag detected; stopping training after epoch {epoch}.")
            break

    torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
    print("✅ Training complete — model saved!")


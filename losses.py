# losses.py
import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    # ensure target is integer long tensor for one_hot
    if not target.dtype == torch.long:
        target = target.long()
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0,4,1,2,3).float()
    intersection = (pred * target_onehot).sum(dim=(2,3,4))
    union = pred.sum(dim=(2,3,4)) + target_onehot.sum(dim=(2,3,4))
    dice = (2*intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

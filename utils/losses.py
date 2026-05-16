import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Implement multi-class Dice Loss
        num_classes = logits.size(1)
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        
        dice_loss = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return (1 - dice_loss).mean()

def get_criterion(loss_type="CrossEntropyLoss"):
    if loss_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_type == "DiceLoss":
        return DiceLoss()
    # Add other losses like FocalLoss here if needed
    else:
        raise ValueError(f"Loss type {loss_type} not supported.")

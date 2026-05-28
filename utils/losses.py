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
        true_1_hot = torch.eye(num_classes, device=logits.device)[targets.squeeze(1) if targets.dim() == 4 else targets]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, targets.ndimension() + 1 if targets.dim() == 3 else targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        
        dice_loss = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return (1 - dice_loss).mean()

class JointCEDiceLoss(nn.Module):
    def __init__(self, num_classes=9, device=None, smooth=1.0, ce_weight=None):
        super(JointCEDiceLoss, self).__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + dice_loss

def get_criterion(loss_type="CrossEntropyLoss", num_classes=9, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if loss_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_type == "DiceLoss":
        return DiceLoss()
    elif loss_type == "JointCEDiceLoss":
        # Configure class weights dynamically
        # Class Index mapping for 9 classes:
        # 0: background (1.0)
        # 1: continuous white (5.0)
        # 2: continuous yellow (5.0)
        # 3: dashed (5.0)
        # 4: double continuous yellow (5.0)
        # 5: main-lane (6.0)
        # 6: other-lane (6.0)
        # 7: turn-lane (6.0)
        # 8: vehicle (1.5)
        if num_classes == 9:
            ce_weight = torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 1.5], dtype=torch.float, device=device)
            print(f"Applying Class Weights in Loss: {ce_weight.cpu().numpy()}")
        else:
            ce_weight = None
            
        return JointCEDiceLoss(num_classes=num_classes, device=device, ce_weight=ce_weight)
    else:
        raise ValueError(f"Loss type {loss_type} not supported.")


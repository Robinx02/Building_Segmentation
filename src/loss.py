#seperate loss file to fetech from
import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    def __init__(self, smooth = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self , logits , targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs_flat * targets).sum()
        dice_score = 1 - (2. * intersection + self.smooth) / (probs_flat.sum() + targets.sum() + self.smooth)
        return dice_score

class BCEDiceLoss(nn.Module):
    def __init__(self , bce_weights = 0.5 , smooth = 1.0):
        super().__init__()
        self.bce_weights = bce_weights
        self.dice = DiceLoss(smooth = smooth)
    
    def forward(self , logits ,targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice(logits ,targets)
        return self.bce_weights * bce_loss  + (1 - self.bce_weights) *dice
        
class IoULoss(nn.Module):
    def __init__(self , smooth =1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self , logits , targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (probs_flat * targets).sum()
        union = probs_flat.sum() + targets.sum() - intersection

        return 1- (intersection + self.smooth) / (union + self.smooth)


class BCEIoULoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou = IoULoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        iou = self.iou(logits, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * iou

def boundary_band(targets , width=2):

    k = 2* width +1
    with torch.no_grad():
        dil = F.max_pool2d(targets , k , stride =1 , padding = width)
        ero = -F.max_pool2d(-targets , k , stride = 1 , padding = width)

        return (dil - ero).clamp(0,1)


class BoundaryWeightedBCEDiceLoss(nn.Module):

    def __init__(self , edge_weight = 5.0 , edge_width =2 , bce_weight = 0.5 ,smooth =1.0):
        super().__init__()
        self.edge_weight , self.edge_width = edge_weight , edge_width
        self.bce_weight , self.smooth = bce_weight , smooth


    def forward(self , logits , targets):
        w = 1.0 + self.edge_weight * boundary_band(targets, self.edge_width)
        bce = F.binary_cross_entropy_with_logits(logits , targets ,reduction = "none")
        bce = (bce * w).sum() / w.sum()

        p = torch.sigmoid(logits)
        inter = (w * p * targets).sum()
        dice = 1 - (2.0 * inter + self.smooth) / ((w * p).sum() + (w * targets).sum() + self.smooth)
        
        return self.bce_weight * bce + (1 - self.bce_weight) * dice  

def get_loss(name, edge_weight=5.0, edge_width=2):
    if name == "bce_dice":
        return BCEDiceLoss()
    if name == "bce_iou":
        return BCEIoULoss()
    if name == "boundary":
        return BoundaryWeightedBCEDiceLoss(edge_weight=edge_weight, edge_width=edge_width)
    raise ValueError(f"unknown loss {name}")

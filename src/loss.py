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
        bce_loss = F.cross_entropy_with_logits(logits , targets)
        dice = self.dice(logits ,targets)
        return self.bace_weights * bce_loss  + (1 - self.bce_weights) *dice
        


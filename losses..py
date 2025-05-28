import torch
import torch.nn as nn

class BoundaryComboLoss(nn.Module):
    def __init__(self):
        super(BoundaryComboLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = 1 - (2. * (pred * target).sum() + 1) / (pred.sum() + target.sum() + 1)
        return bce_loss + dice_loss
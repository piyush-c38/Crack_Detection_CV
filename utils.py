import torch

def iou_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def dice_score(pred, target):
    pred = (pred > 0.5).float()
    return (2 * (pred * target).sum() + 1e-6) / (pred.sum() + target.sum() + 1e-6)
import torch

_bce = torch.nn.BCEWithLogitsLoss() #Loss function that applies sigmoid and calculates BCE

def bce_loss(pred, mask):
    loss = _bce(pred,mask)
    return loss

def dice_loss(pred, mask):
    epsilon = 10e-5
    prob = torch.sigmoid(pred)
    loss = 1 - (2*torch.sum(prob*mask) + epsilon) / (torch.sum(prob) + torch.sum(mask) + epsilon)
    return loss

def combo_loss(pred, mask, bce_weight=0.5, dice_weight=0.5):
    loss = bce_weight*bce_loss(pred,mask) + dice_weight*dice_loss(pred,mask)
    return loss
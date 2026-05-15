import torch
from torch.nn.functional import binary_cross_entropy_with_logits


def weighted_bce_loss(logits, target, pos_weight=10.0):
    target = target.to(device=logits.device, dtype=torch.float32).view(-1, 1)
    weight = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    
    loss =  binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    return loss

def bce_fn_penalty_loss(logits, target, pos_weight=10.0, fn_penalty_weight=1.0):
    target = target.to(device=logits.device, dtype=torch.float32).view(-1, 1)
    bce = weighted_bce_loss(logits, target, pos_weight=pos_weight)

    prob = torch.sigmoid(logits)
    soft_fn = torch.sum((1.0 - prob) * target)
    fn_penalty = soft_fn / (target.sum() + 1e-8)

    loss = bce + fn_penalty_weight* fn_penalty
    return loss

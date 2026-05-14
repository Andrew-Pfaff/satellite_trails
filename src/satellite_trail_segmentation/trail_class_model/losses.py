import torch
from torch.nn.functional import binary_cross_entropy_with_logits


def weighted_bce_loss(logits, target, pos_weight=10.0):
    target = target.to(device=logits.device, dtype=torch.float32).view(-1, 1)
    weight = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return binary_cross_entropy_with_logits(logits, target, pos_weight=weight)


def recall_combo_loss(logits, target, pos_weight=10.0, recall_weight=1.0, epsilon=1e-8):
    target = target.to(device=logits.device, dtype=torch.float32).view(-1, 1)
    bce = weighted_bce_loss(logits, target, pos_weight=pos_weight)

    probabilities = torch.sigmoid(logits)
    true_positive = torch.sum(probabilities * target)
    false_negative = torch.sum((1.0 - probabilities) * target)
    recall = true_positive / (true_positive + false_negative + epsilon)

    return bce - recall_weight * recall

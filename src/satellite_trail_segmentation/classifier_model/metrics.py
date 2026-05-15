import torch


def batch_metrics(logits, target, threshold=0.3, epsilon=1e-8):
    target = target.to(device=logits.device, dtype=torch.float32).view(-1, 1)
    pred = (torch.sigmoid(logits) >= threshold).float()
    true_positive = torch.sum(pred * target).item()
    false_positive = torch.sum(pred * (1.0 - target)).item()
    false_negative = torch.sum((1.0 - pred) * target).item()
    true_negative = torch.sum((1.0 - pred) * (1.0 - target)).item()

    recall = true_positive / (true_positive + false_negative + epsilon)
    precision = true_positive / (true_positive + false_positive + epsilon)
    fnr = false_negative / (true_positive + false_negative + epsilon)
    fpr = false_positive / (false_positive + true_negative + epsilon)

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "recall": recall,
        "precision": precision,
        "fnr": fnr,
        "fpr": fpr,
    }

import torch


def batch_metrics(logits, target, threshold=0.3, epsilon=1e-8):
    """
    Calculates batch-level binary classification metrics from classifier logits.

    Converts logits into binary predictions, counts confusion-matrix elements, and derives recall, precision, false-negative rate, and false-positive rate for the batch.

    Args:
        logits (torch.Tensor): Raw classifier outputs with shape (batch_size, 1).
        target (torch.Tensor): Ground truth binary labels with the same batch shape.
        threshold (float, optional): Probability threshold used to binarize predictions. Defaults to 0.3.
        epsilon (float, optional): Small constant used to avoid division by zero. Defaults to 1e-8.

    Returns:
        dict: A dictionary containing the calculated metrics and absolute counts:
            - "true_positive" (float): Total true positive count.
            - "false_positive" (float): Total false positive count.
            - "false_negative" (float): Total false negative count.
            - "true_negative" (float): Total true negative count.
            - "recall" (float): True positive rate.
            - "precision" (float): True positive rate among predicted positives.
            - "fnr" (float): False negative rate.
            - "fpr" (float): False positive rate.
    """

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

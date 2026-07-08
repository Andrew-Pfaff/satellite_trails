import numpy as np
import torch

def init_conf_counts():
    """
    Creates an empty binary confusion-count dictionary.

    Returns:
        counts (dict): Dictionary containing zeroed true-positive, false-positive, false-negative, and true-negative counts.
    """

    return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}


def update_conf_counts_batch(total_counts, batch_counts):
    """
    Adds one batch of binary confusion counts into an aggregate count dictionary.

    Args:
        total_counts (dict): Running aggregate count dictionary containing ``tp``, ``fp``, ``fn``, and ``tn``.
        batch_counts (dict): Batch count dictionary containing ``tp``, ``fp``, ``fn``, and ``tn``.

    Returns:
        total_counts (dict): The updated aggregate count dictionary.
    """

    total_counts["tp"] += batch_counts["tp"]
    total_counts["fp"] += batch_counts["fp"]
    total_counts["fn"] += batch_counts["fn"]
    total_counts["tn"] += batch_counts["tn"]

    return total_counts


def conf_counts_from_logits(logits, target, threshold=0.5):
    """
    Converts raw logits into binary predictions and counts confusion-matrix elements.

    Applies a sigmoid activation to model logits, thresholds the resulting probabilities, flattens predictions and targets, then counts true positives, false positives, false negatives, and true negatives. This works for both patch-level classifier outputs and pixel-level segmentation outputs.

    Args:
        logits (torch.Tensor): Raw model outputs. Classifier logits may have shape ``(batch_size, 1)`` and segmentation logits may have shape ``(batch_size, 1, height, width)``.
        target (torch.Tensor): Binary target tensor with the same shape as ``logits`` or a compatible flattened shape.
        threshold (float, optional): Probability threshold used after sigmoid activation. Defaults to 0.5.

    Returns:
        counts (dict): Dictionary containing ``tp``, ``fp``, ``fn``, and ``tn`` counts.
    """

    target = target.to(device=logits.device, dtype=torch.float32).view(-1)
    pred = (torch.sigmoid(logits).view(-1) >= threshold).float()

    pixels = target.numel()
    total_pred_pos = torch.sum(pred).item()
    total_target_pos = torch.sum(target).item()

    tp = torch.sum(pred * target).item()
    fp = total_pred_pos - tp
    fn = total_target_pos - tp
    tn = pixels - tp - fn - fp

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def conf_counts_from_arrays(pred_bin, target):
    """
    Counts binary confusion-matrix elements from already-binarized predictions and targets.

    Args:
        pred_bin (np.ndarray or torch.Tensor): Binary prediction array or tensor.
        target (np.ndarray or torch.Tensor): Binary target array or tensor.

    Returns:
        counts (dict): Dictionary containing ``tp``, ``fp``, ``fn``, and ``tn`` counts.
    """

    pred_bin = np.asarray(pred_bin).astype(bool).flatten()
    target = np.asarray(target).astype(bool).flatten()

    tp = np.sum(pred_bin & target)
    fp = np.sum(pred_bin & ~target)
    fn = np.sum(~pred_bin & target)
    tn = np.sum(~pred_bin & ~target)

    return {"tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}


def metrics_from_conf_counts(counts, epsilon=1e-8):
    """
    Calculates binary classification and segmentation metrics from aggregate confusion counts.

    Args:
        counts (dict): Dictionary containing ``tp``, ``fp``, ``fn``, and ``tn`` counts.
        epsilon (float, optional): Small numerical value used to avoid division by zero. Defaults to 1e-8.

    Returns:
        metrics (dict): Dictionary containing confusion counts and derived metrics: accuracy, precision, recall, sensitivity, specificity, false-negative rate, false-positive rate, intersection over union, and Dice score.
    """

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    fnr = fn / (tp + fn + epsilon)
    fpr = fp / (fp + tn + epsilon)


    return {"tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "iou": iou,
            "dice": dice,
            "fnr": fnr,
            "fpr": fpr}


def threshold_sweep_from_logits(logits, target, thresholds):
    """
    Calculates binary confusion counts for several probability thresholds.

    Args:
        logits (torch.Tensor): Raw model outputs.
        target (torch.Tensor): Binary target tensor.
        thresholds (iterable): Probability thresholds used after sigmoid activation.

    Returns:
        threshold_counts (dict): Dictionary mapping each threshold to a count dictionary containing ``tp``, ``fp``, ``fn``, and ``tn``.
    """

    threshold_counts = {}
    for threshold in thresholds:
        threshold_counts[threshold] = conf_counts_from_logits(logits, target, threshold=threshold)

    return threshold_counts


def best_threshold_by_metric(threshold_metrics, metric_name):
    """
    Selects the threshold with the highest value for a requested metric.

    Args:
        threshold_metrics (dict): Dictionary mapping thresholds to metric dictionaries.
        metric_name (str): Name of the metric to maximize.

    Returns:
        tuple: A 2-element tuple containing the best threshold and its corresponding metrics dictionary.
    """

    return max(threshold_metrics.items(), key=lambda item: item[1][metric_name])


def specificity_with_recall_penalty(metrics, min_recall, penalty):
    """
    Calculates specificity after penalizing recall below a minimum target.

    Args:
        metrics (dict): Metric dictionary containing ``specificity`` and ``recall``.
        min_recall (float): Desired minimum recall.
        penalty (float): Penalty multiplier for recall shortfall.

    Returns:
        float: Penalized specificity score.
    """

    specificity = metrics['specificity']
    recall = metrics['recall']
    
    sub_recall_value = max(0.0, min_recall - recall)

    return specificity - sub_recall_value * penalty


def best_threshold_by_penalized_specificity(threshold_metrics, min_recall, penalty):
    """
    Selects the threshold with the highest penalized specificity.

    Args:
        threshold_metrics (dict): Dictionary mapping thresholds to metric dictionaries.
        min_recall (float): Desired minimum recall.
        penalty (float): Penalty multiplier for recall shortfall.

    Returns:
        tuple: A 2-element tuple containing the best threshold and its metrics dictionary.
    """

    return max(threshold_metrics.items(), key=lambda item: specificity_with_recall_penalty(item[1], min_recall, penalty))

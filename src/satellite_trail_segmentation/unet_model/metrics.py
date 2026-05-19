import numpy as np
from sklearn.metrics import roc_curve, auc


def accuracy_metrics(pred_bin, mask):
    """
    Calculates pixel-level classification and segmentation performance metrics.

    Computes basic confusion matrix elements (TP, TN, FP, FN) and derived evaluation metrics from the predictions and ground-truth masks.

    Args:
        pred_bin (np.ndarray): Binary segmentation predictions thresholded to contain only integer classes (e.g., 0 for background, 1 for trail). 
        mask (np.ndarray): Ground truth binary segmentation mask containing integer values (0 or 1).

    Returns:
        dict: A dictionary containing the calculated metrics and absolute pixel counts:
            - "accuracy" (float): Proportion of correctly classified pixels.
            - "precision" (float): True positive rate among predicted positives.
            - "sensitivity" (float): True positive rate (Recall / Hit rate).
            - "specificity" (float): True negative rate (Selectivity).
            - "iou" (float): Intersection over Union (Jaccard Index).
            - "dice" (float): Dice Similarity Coefficient (F1-score equivalent).
            - "tp" (int): Total true positive pixel count.
            - "tn" (int): Total true negative pixel count.
            - "fp" (int): Total false positive pixel count.
            - "fn" (int): Total false negative pixel count.
            - "num_pix" (int): Total number of pixels evaluated.
    """
    
    pred_trail = (pred_bin > 0).astype(bool).flatten()
    mask_trail = (mask > 0).astype(bool).flatten()
    pred_dark = (pred_bin == 0).astype(bool).flatten()
    mask_dark = (mask == 0).astype(bool).flatten()

    total_pixels = len(pred_trail)
    
    tp = np.sum(pred_trail & mask_trail)
    tn = np.sum(pred_dark & mask_dark)
    fp = np.sum(pred_trail & mask_dark)
    fn = np.sum(pred_dark & mask_trail)


    epsilon = 1e-8
    accuracy = (tp + tn) / total_pixels
    precision = (tp) / (tp + fp + epsilon)
    sensitivity = (tp) / (tp + fn + epsilon)
    specificity = (tn) / (tn + fp + epsilon)
    dice = (2*tp) / (2*tp + fp + fn + epsilon)
    iou = (tp) / (tp + fp + fn + epsilon)
    

    results = {"accuracy": accuracy,
               "precision": precision,
               "sensitivity": sensitivity,
               "specificity": specificity,
               "iou": iou,
               "dice": dice,
               "tp": tp,
               "tn": tn,
               "fp": fp,
               "fn": fn,
               "num_pix": total_pixels}

    return results


def get_roc_auc_data(pred_prob, mask):
    """
    Computes the Receiver Operating Characteristic (ROC) curve metrics and thresholds.

    Flattens continuous prediction probabilities and targets to calculate the area under the curve (AUC-ROC) and extracts the optimal probability threshold maximizing TPR - FPR.

    Args:
        pred_prob (np.ndarray): Continuous prediction probability maps output by the network (values between 0.0 and 1.0).
        mask (np.ndarray): Ground truth binary segmentation mask containing integer values (0 or 1). 
    Returns:
        tuple: A 5-element tuple containing:
            - fpr (np.ndarray): False Positive Rates for increasingly strict thresholds.
            - tpr (np.ndarray): True Positive Rates for increasingly strict thresholds.
            - thresholds (np.ndarray): Probability thresholds used to compute fpr and tpr.
            - optimal_threshold (float): The threshold that maximizes Youden's J statistic.
            - roc_auc (float): The calculated Area Under the ROC Curve (AUC).
    """

    mask_f = mask.flatten()
    pred_prob_f = pred_prob.flatten()
    
    fpr, tpr, thresholds = roc_curve(mask_f, pred_prob_f)
    idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[idx]

    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, optimal_threshold, roc_auc
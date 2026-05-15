import numpy as np
from sklearn.metrics import roc_curve, auc


def accuracy_metrics(pred_bin, mask):
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
    mask_f = mask.flatten()
    pred_prob_f = pred_prob.flatten()
    
    fpr, tpr, thresholds = roc_curve(mask_f, pred_prob_f)
    idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[idx]

    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, optimal_threshold, roc_auc
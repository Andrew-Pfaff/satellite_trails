import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from satellite_trail_segmentation.unet_model.losses import combo_loss
from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.unet_model.metrics import accuracy_metrics, get_roc_auc_data


def evaluate_patches(model, h5_path, split_type, batch_size, subsample_fraction=0.01):
    """
    Evaluates all patches within a custom h5 path that are within the specified split type.
    
    Computes a global combo-loss, and uses a randomly subsampled fraction of pixels across the dataset to calculate ROC-AUC and binary classification metrics using an optimal threshold.
    
    Args:
        model (torch.nn.Module): Trained UNet model to evaluate.
        h5_path (str): Path to the h5 file containing the dataset.
        split_type (str): Type of data split to evaluate. Must be "train", 
            "val", or "test".
        batch_size (int): Number of patches per batch.
        subsample_fraction (float, optional): Percentage of pixels to randomly 
            sample per batch for memory-efficient ROC-AUC calculations. 
            Defaults to 0.01.

    Returns:
        tuple: A tuple containing:
            - test_loss (float): The average combo-loss (Dice + BCE) across 
              all batches.
            - metrics (dict): Subsampled pixel-level evaluation metrics containing:
                * "accuracy": Global classification accuracy.
                * "precision": Precision score.
                * "sensitivity": Recall / True Positive Rate.
                * "specificity": True Negative Rate.
                * "iou": Intersection over Union.
                * "dice": Dice similarity coefficient.
                * "tp": Total true positives.
                * "tn": Total true negatives.
                * "fp": Total false positives.
                * "fn": Total false negatives.
                * "num_pix": Total number of evaluated pixels.
                * "roc_auc": Area under the ROC curve.
                * "optimal_threshold": The threshold used to binarize predictions.
            - roc_data (dict): Arrays for plotting the ROC curve, containing:
                * "fpr" (numpy.ndarray): False Positive Rates.
                * "tpr" (numpy.ndarray): True Positive Rates.
                * "thresholds" (numpy.ndarray): Thresholds evaluated for the curve.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = H5PatchDataset(h5_path, split=split_type, return_metadata=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    with torch.no_grad():
        model.eval()
        test_loss = 0
        sampled_pred = []
        sampled_mask = []
        for images, masks, metadata in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = combo_loss(logits, masks)

            test_loss += 1/len(loader) * loss.item()

            # --- NEW METRIC COLLECTION LOGIC ---
            # 1. Apply sigmoid, move to CPU, and flatten into 1D arrays
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            masks_np = masks.cpu().numpy().flatten()
            
            # 2. Randomly sample a fraction of the pixels to save RAM
            num_samples = int(len(probs) * subsample_fraction)
            subsample_idx = np.random.choice(len(probs), size=num_samples, replace=False)
            
            # 3. Store the subsampled arrays
            sampled_pred.append(probs[subsample_idx])
            sampled_mask.append(masks_np[subsample_idx])
    
    
    # --- SAMPLED METRIC CALCULATION ---
    # Concatenate all subsampled batches into single global arrays
    sampled_pred = np.concatenate(sampled_pred)
    sampled_mask = np.concatenate(sampled_mask)

    # Calculate global ROC and find the optimal threshold
    fpr, tpr, thresholds, optimal_threshold, roc_auc = get_roc_auc_data(sampled_pred, sampled_mask)

    # Binarize the predictions using that optimal threshold
    pred_bin = (sampled_pred > optimal_threshold).astype(np.uint8)

    # Calculate final accuracy metrics (IoU, Dice, Precision, etc.)
    metrics = accuracy_metrics(pred_bin, sampled_mask)
    
    # Store the ROC info inside the metrics dict for convenience
    metrics["roc_auc"] = roc_auc
    metrics["optimal_threshold"] = optimal_threshold

    # Package the raw curve data for your separate plotting script
    roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    return test_loss, metrics, roc_data


def image_threshold(image, threshold=0.5):
    """
    Thresholds an image into binary (0 or 255), using a single threshold value.

    Args:
        image (np.ndarray): 2D image array
        threshold (float): Value for which values above are set to 255 and values equal or below are set to 0.

    Returns:
        binary_image (np.ndarray): Binarized 2D image array
    """

    binary_image = (image > threshold)
    binary_image = binary_image.astype(np.uint8)*255
    return binary_image


def recreate_full_field_pred(model, h5_path, split_type, source_index, batch_size=1, patch_dim=528):
    """
    Reassembles individual patches back into a single full-field image, complete with their corresponding predictions and ground truth masks.

    Iterates through patches belonging to a specific source image, runs them through the model to get probability maps, and places them back into their original spatial coordinates.

    Args:
        model (torch.nn.Module): Trained UNet model used to generate patch predictions.
        h5_path (str): Path to the h5 file containing the dataset and metadata.
        split_type (str): Type of data split to evaluate. Must be "train", "val", or "test".
        source_index (int): The unique index identifier of the full-field source image to reconstruct.
        batch_size (int, optional): Number of patches per batch during inference. Defaults to 1.
        patch_dim (int, optional): Spatial dimension (height and width) of the square patches. Defaults to 528.

    Returns:
        tuple: A tuple containing:
            - full_image (numpy.ndarray): Reconstructed 2D/3D original image array.
            - full_pred (numpy.ndarray): Reconstructed 2D probability map array containing values from 0.0 to 1.0.
            - full_mask (numpy.ndarray): Reconstructed 2D ground truth binary mask array.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with h5py.File(h5_path, "r") as f:
        full_shape = tuple(f.attrs["full_shape"])

    dataset = H5PatchDataset(h5_path, split=split_type, return_metadata=True, source_index=source_index)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    full_image = np.zeros(full_shape, dtype=np.float32)
    full_pred = np.zeros(full_shape, dtype=np.float32)
    full_mask = np.zeros(full_shape, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for images, masks, metadata in loader:
            pred_logits = model(images.to(device))
            preds = torch.sigmoid(pred_logits).squeeze(1).cpu().numpy()

            masks = masks.squeeze(1).numpy()
            images = images.squeeze(1).numpy()
            
            for i in range(len(preds)):
                y0 = metadata["patch_y0"][i].item()
                x0 = metadata["patch_x0"][i].item()
                full_image[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = images[i]
                full_pred[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = preds[i]
                full_mask[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = masks[i]

    return full_image, full_pred, full_mask


def evaluate_full_field_pred(full_pred, full_mask, threshold):
    """
    Evaluates segmentation performance metrics and generates ROC data on a fully reassembled image field.

    Binarizes the continuous probability map using a provided threshold value and evaluates standard segmentation accuracy metrics against the complete ground truth mask.

    Args:
        full_pred (numpy.ndarray): The reassembled full-field continuous probability map array.
        full_mask (numpy.ndarray): The reassembled full-field ground truth binary mask array.
        threshold (float): The cutoff value used to binarize `full_pred` into a discrete pixel mask.

    Returns:
        tuple: A tuple containing:
            - metrics (dict): Full-field pixel-level evaluation metrics containing:
                * "accuracy": Overall classification accuracy.
                * "precision": Precision score.
                * "sensitivity": Recall / True Positive Rate.
                * "specificity": True Negative Rate.
                * "iou": Intersection over Union.
                * "dice": Dice similarity coefficient.
                * "tp": Total true positives.
                * "tn": Total true negatives.
                * "fp": Total false positives.
                * "fn": Total false negatives.
                * "num_pix": Total number of pixels evaluated.
                * "roc_auc": Area under the ROC curve calculated from the full-field array.
                * "optimal_threshold": Theoretically optimal threshold found via ROC analysis.
            - roc_data (dict): Arrays for plotting the ROC curve, containing:
                * "fpr" (numpy.ndarray): False Positive Rates.
                * "tpr" (numpy.ndarray): True Positive Rates.
                * "thresholds" (numpy.ndarray): Evaluated thresholds along the curve.
    """

    fpr, tpr, thresholds, optimal_threshold, roc_auc = get_roc_auc_data(full_pred, full_mask)
    
    pred_bin = image_threshold(full_pred, threshold)
    metrics = accuracy_metrics(pred_bin, full_mask)
    
    metrics["roc_auc"] = roc_auc
    metrics["optimal_threshold"] = optimal_threshold
    roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    return metrics, roc_data
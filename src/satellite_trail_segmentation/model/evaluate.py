import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from satellite_trail_segmentation.model.losses import combo_loss
from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.model.metrics import accuracy_metrics, get_roc_auc_data


def evaluate_patches(model, h5_path, split_type, batch_size, subsample_fraction=0.01):
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
    sampled_masks = np.concatenate(sampled_masks)

    # Calculate global ROC and find the optimal threshold
    fpr, tpr, thresholds, optimal_threshold, roc_auc = get_roc_auc_data(sampled_pred, sampled_masks)

    # Binarize the predictions using that optimal threshold
    pred_bin = (sampled_pred > optimal_threshold).astype(np.uint8)

    # Calculate final accuracy metrics (IoU, Dice, Precision, etc.)
    metrics = accuracy_metrics(pred_bin, sampled_masks)
    
    # Store the ROC info inside the metrics dict for convenience
    metrics["roc_auc"] = roc_auc
    metrics["optimal_threshold"] = optimal_threshold

    # Package the raw curve data for your separate plotting script
    roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    return test_loss, metrics, roc_data


def image_threshold(image, threshold=0.5):
    binary_image = (image > threshold)
    binary_image = binary_image.astype(np.uint8)*255
    return binary_image


def recreate_full_field_pred(model, h5_path, split_type, source_index, batch_size=1, patch_dim=528):
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


def evaluate_full_field_pred(full_image, full_pred, full_mask, threshold):
    fpr, tpr, thresholds, optimal_threshold, roc_auc = get_roc_auc_data(full_pred, full_mask)
    
    pred_bin = image_threshold(full_pred, threshold)
    metrics = accuracy_metrics(pred_bin, full_mask)
    
    metrics["roc_auc"] = roc_auc
    metrics["optimal_threshold"] = optimal_threshold
    roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    return metrics, roc_data, full_image, full_pred, full_mask
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryROC
from sklearn.metrics import auc

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.ml_utils.metrics import init_conf_counts, update_conf_counts_batch, conf_counts_from_logits, metrics_from_conf_counts, calculate_patch_wise_metrics


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


def evaluate_dataset_unet(model, h5_path, split_type, pred_thresholds=None, batch_size=1, ):
    if pred_thresholds is None:
        pred_thresholds = [0.5]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = H5PatchDataset(h5_path, split=split_type, return_masks=True, return_metadata=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    threshold_counts = {t: init_conf_counts() for t in pred_thresholds}

    roc_metric = BinaryROC(thresholds=500).to(device)

    threshold_patch_histories = {t: [] for t in pred_thresholds}

    model.eval()
    with torch.no_grad():
        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            prediction = model(images)
            probs = torch.sigmoid(prediction)

            roc_metric.update(probs.flatten(), masks.flatten().long())
            
            for t in pred_thresholds:
                for i in range(prediction.size(0)):
                    single_patch_logits = prediction[i : i + 1]
                    single_patch_mask = masks[i : i + 1]

                    patch_dict = conf_counts_from_logits(logits=single_patch_logits, target=single_patch_mask, threshold=t)

                    threshold_patch_histories[t].append(patch_dict)

                    update_conf_counts_batch(threshold_counts[t], patch_dict)

    metrics_counts = {t: metrics_from_conf_counts(counts) for t, counts in threshold_counts.items()}

    fpr_tensor, tpr_tensor, thresholds_tensor = roc_metric.compute()

    fpr = fpr_tensor.cpu().numpy()
    tpr = tpr_tensor.cpu().numpy()
    thresholds = thresholds_tensor.cpu().numpy()

    idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[idx]

    sort_idx = np.argsort(fpr)
    fpr = fpr[sort_idx]
    tpr = tpr[sort_idx]
    thresholds = thresholds[sort_idx]

    roc_auc = auc(fpr, tpr)

    roc_metric.reset()

    patch_metrics_counts = {t: calculate_patch_wise_metrics(history_list) for t, history_list in threshold_patch_histories.items()}

    return metrics_counts, patch_metrics_counts, fpr, tpr, thresholds, optimal_threshold, roc_auc
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.ml_utils.metrics import (
    init_conf_counts, 
    update_conf_counts_batch, 
    conf_counts_from_logits,
    conf_counts_from_arrays,
    metrics_from_conf_counts
)


def _predict(logits, threshold=0.3):
    """
    Converts classifier logits into binary predictions using a threshold.

    Args:
        logits (torch.Tensor): Raw classifier outputs with shape (batch_size, 1).
        threshold (float, optional): Probability threshold used to binarize predictions. Defaults to 0.3.

    Returns:
        torch.Tensor: Binary predictions as integer tensors with the same batch shape.
    """

    probabilities = torch.sigmoid(logits)
    return (probabilities >= threshold).to(dtype=torch.int64)


def recreate_full_field(model, h5_path, split_type, source_index, batch_size=32, patch_dim=528, threshold=0.5, normalization="source_zscore"):
    """
    Reassembles patch-level classifier predictions into a full-field view for a single source image.

    Iterates through all patches for the requested source image, runs the classifier on each patch, and places the image content, predicted labels, and ground truth labels back into their original spatial positions.

    Args:
        model (torch.nn.Module): Trained classifier model used to generate patch predictions.
        h5_path (str): Path to the h5 file containing the dataset and metadata.
        split_type (str): Type of data split to evaluate. Must be "train", "val", or "test".
        source_index (int): The unique index identifier of the full-field source image to reconstruct.
        batch_size (int, optional): Number of patches per batch during inference. Defaults to 32.
        patch_dim (int, optional): Spatial dimension (height and width) of the square patches. Defaults to 528.
        threshold (float, optional): Probability cutoff used to binarize classifier outputs. Defaults to 0.5.
        normalization (str, optional): Dataset normalization mode. Defaults to "source_zscore".

    Returns:
        tuple: A tuple containing:
            - full_image (numpy.ndarray): Reconstructed image array.
            - full_pred (numpy.ndarray): Reconstructed binary prediction array.
            - full_mask (numpy.ndarray): Reconstructed ground truth label array.
            - full_overlay (numpy.ndarray): RGB overlay showing patch predictions and correctness.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with h5py.File(h5_path, "r") as f:
        full_shape = tuple(f.attrs["full_shape"])

    dataset = H5PatchDataset(h5_path, split=split_type, return_metadata=True, return_masks=False, source_index=source_index, normalization=normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    full_image = np.zeros(full_shape, dtype=np.float32)
    full_pred = np.zeros(full_shape, dtype=np.float32)
    full_mask = np.zeros(full_shape, dtype=np.float32)
  
    square_size=100
    full_overlay = np.zeros((*full_shape, 3), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for images, metadata in loader:
            logits = model(images.to(device))
            preds = _predict(logits, threshold=threshold).squeeze(1).cpu().numpy().astype(np.float32)

            images = images.squeeze(1).numpy()

            for i in range(len(preds)):
                y0 = metadata["patch_y0"][i].item()
                x0 = metadata["patch_x0"][i].item()
                full_image[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = images[i]
                full_pred[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = preds[i]
                full_mask[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = metadata["patch_has_trail"][i].item()
                


                bg_color = np.array([1.0, 1.0, 1.0]) if preds[i] == 1 else np.array([0.0, 0.0, 0.0])
                full_overlay[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = bg_color
   
                mask_val = metadata["patch_has_trail"][i].item()
                center_color = np.array([0.0, 1.0, 0.0]) if preds[i] == mask_val else np.array([1.0, 0.0, 0.0])

                y_start = y0 + (patch_dim - square_size) // 2
                x_start = x0 + (patch_dim - square_size) // 2

                full_overlay[y_start : y_start + square_size, x_start : x_start + square_size] = center_color


    return full_image, full_pred, full_mask, full_overlay


def evaluate_dataset_classifier(model, h5_path, split_type, pred_thresholds=None, batch_size=4, normalization="source_zscore", num_workers=0):
    """
    Evaluates a classifier over one HDF5 split and threshold sweep.

    Args:
        model (torch.nn.Module): Trained classifier model.
        h5_path (str): Path to the HDF5 patch dataset.
        split_type (str): Split to evaluate: "train", "val", or "test".
        pred_thresholds (list, optional): Probability thresholds for binary metrics.
            Defaults to [0.5].
        batch_size (int): Number of patches per inference batch. Defaults to 4.
        normalization (str): Dataset normalization mode. Defaults to "source_zscore".
        num_workers (int): DataLoader worker count. Defaults to 0.

    Returns:
        tuple: Threshold-level metrics and per-threshold image-wise confusion counts.
    """

    if pred_thresholds is None:
        pred_thresholds = [0.5]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = H5PatchDataset(h5_path, split=split_type, return_metadata=True, return_masks=False, normalization=normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    threshold_counts = {t: init_conf_counts() for t in pred_thresholds}
    image_wise_counts = {t: {} for t in pred_thresholds}

    model.eval()
    with torch.no_grad():
        for images, metadata in loader:
            images = images.to(device)
            targets = metadata["patch_has_trail"].to(device=device, dtype=torch.float32).view(-1, 1)

            prediction = model(images)
        
            for t in pred_thresholds:
                #Batch-wise counts
                batch_counts = conf_counts_from_logits(logits=prediction, target= targets, threshold=t)
                threshold_counts[t] = update_conf_counts_batch(threshold_counts[t], batch_counts)
            
                #Image-wise counts
                preds_bin = (torch.sigmoid(prediction) >= t).cpu().numpy().flatten()
                targets_np = targets.cpu().numpy().flatten()
                source_indices = metadata['source_index']
                for i, img_id in enumerate(source_indices):
                    img_key = int(img_id)

                    single_pred = preds_bin[i]
                    single_target = targets_np[i]
                
                    patch_counts = conf_counts_from_arrays(single_pred, single_target)

                    if img_key not in image_wise_counts[t]:
                        image_wise_counts[t][img_key] = init_conf_counts()
                    
                    image_wise_counts[t][img_key] = update_conf_counts_batch(image_wise_counts[t][img_key], patch_counts)


    metrics = {t: metrics_from_conf_counts(counts) for t, counts in threshold_counts.items()}

    return metrics, image_wise_counts

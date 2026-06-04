import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from satellite_trail_segmentation.ml_utils.loss_functions import combo_loss
from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.ml_utils.metrics import init_conf_counts, update_conf_counts_batch, conf_counts_from_logits, metrics_from_conf_counts


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


def evaluate_dataset_unet(model, h5_path, split_type, pred_thresholds=None, batch_size=1):
    if pred_thresholds is None:
        pred_thresholds = [0.5]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = H5PatchDataset(h5_path, split=split_type, return_masks=True, return_metadata=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    threshold_counts = {t: init_conf_counts() for t in pred_thresholds}
    patches_processed = 0

    model.eval()
    with torch.no_grad():
        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            prediction = model(images)

            for t in pred_thresholds:
                batch_counts = conf_counts_from_logits(logits=prediction, target=masks, threshold=t)
                update_conf_counts_batch(threshold_counts[t], batch_counts)

            patches_processed += batch_size
            if patches_processed % 100 == 0:
                print(f"{patches_processed} have been processed.")
    

    metrics_counts = {t: metrics_from_conf_counts(counts) for t, counts in threshold_counts.items()}

    return metrics_counts
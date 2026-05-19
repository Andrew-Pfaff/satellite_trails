import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from satellite_trail_segmentation.data.dataset import H5PatchDataset


def predict(logits, threshold=0.3):
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


def recreate_full_field(model, h5_path, split_type, source_index, batch_size=32, patch_dim=528, threshold=0.3):
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
        threshold (float, optional): Probability cutoff used to binarize classifier outputs. Defaults to 0.3.

    Returns:
        tuple: A tuple containing:
            - full_image (numpy.ndarray): Reconstructed image array.
            - full_pred (numpy.ndarray): Reconstructed binary prediction array.
            - full_mask (numpy.ndarray): Reconstructed ground truth label array.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with h5py.File(h5_path, "r") as f:
        full_shape = tuple(f.attrs["full_shape"])

    dataset = H5PatchDataset(h5_path, split=split_type, return_metadata=True, return_masks=False, source_index=source_index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    full_image = np.zeros(full_shape, dtype=np.float32)
    full_pred = np.zeros(full_shape, dtype=np.float32)
    full_mask = np.zeros(full_shape, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for images, metadata in loader:
            logits = model(images.to(device))
            preds = predict(logits, threshold=threshold).squeeze(1).cpu().numpy().astype(np.float32)

            images = images.squeeze(1).numpy()

            for i in range(len(preds)):
                y0 = metadata["patch_y0"][i].item()
                x0 = metadata["patch_x0"][i].item()
                full_image[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = images[i]
                full_pred[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = preds[i]
                full_mask[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = metadata["patch_has_trail"][i].item()

    return full_image, full_pred, full_mask

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from satellite_trail_segmentation.model.losses import combo_loss
from satellite_trail_segmentation.data.dataset import H5PatchDataset


def evaluate_patches(model, h5_path, split_type, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = H5PatchDataset(h5_path, split=split_type, return_metadata=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    with torch.no_grad():
        model.eval()
        test_loss = 0
        for images, masks, metadata in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = combo_loss(logits, masks)

            test_loss += 1/len(loader) * loss.item()

    return test_loss


def evaluate_full_fields(model, h5_path, split_type):
    dataset = H5PatchDataset(h5_path, split=split_type)


def image_threshold(image, threshold=0.5):
    binary_image = (image > threshold)
    binary_image = binary_image.astype(np.uint8)*255
    return binary_image


def recreate_full_field(model, h5_path, split_type, source_index, batch_size=1, patch_dim=528):
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
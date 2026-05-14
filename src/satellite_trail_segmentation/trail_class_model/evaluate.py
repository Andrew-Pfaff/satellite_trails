import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from satellite_trail_segmentation.data.dataset import H5PatchDataset


def predict(logits, threshold=0.3):
    probabilities = torch.sigmoid(logits)
    return (probabilities >= threshold).to(dtype=torch.int64)


def recreate_full_field(model, h5_path, split_type, source_index, batch_size=32, patch_dim=528, threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with h5py.File(h5_path, "r") as f:
        full_shape = tuple(f.attrs["full_shape"])

    dataset = H5PatchDataset(h5_path, split=split_type, return_metadata=True, source_index=source_index)
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

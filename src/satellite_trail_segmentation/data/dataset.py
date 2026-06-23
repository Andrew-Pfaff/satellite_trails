import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from satellite_trail_segmentation.data.augmentation import augment_image

class H5PatchDataset(Dataset):
    """
    PyTorch Dataset for loading image and mask patches from an HDF5 file.

    Supports train/val/test splitting, optional augmentation on trail-containing patches, source-level normalization, and optional metadata and mask returns. Augmentation is only applied to the train split.
    The HDF5 file is opened lazily on the first call to __getitem__ and held open for the lifetime of the dataset.
    
    Attributes:
        h5_path (str): Path to the source HDF5 file.
        split (str): The dataset split type ('train', 'val', or 'test').
        return_metadata (bool): If True, __getitem__ includes a dictionary of patch coordinates.
        return_masks (bool): If True, __getitem__ includes the target segmentation mask tensor.
        augment (bool): Indicates whether data augmentation will be applied during retrieval.
        valid_indices (np.ndarray): Array mapping dataset items back to global HDF5 row indices.
        pos_indices (np.ndarray): Subset of internal indices pointing strictly to trail-containing patches.
        neg_indices (np.ndarray): Subset of internal indices pointing strictly to trail-free patches.
    """

    def __init__(self, h5_path, split='train', return_metadata=False, return_masks=True, source_index=None, augment=False, p_flip=0.1, p_rot=0.1, normalization="source_zscore"):
        """
        Args:
            h5_path (str): Path to the HDF5 file created by create_h5
            split (str): One of 'train', 'val', or 'test'
            return_metadata (bool): If True, returns patch metadata dict. Defaults to False.
            return_masks (bool): If True, returns mask tensor. Defaults to True.
            source_index (int): If provided, restricts dataset to patches from a single source image. Defaults to None.
            augment (bool): If True and split is 'train', applies random augmentation
                to trail-containing patches. Defaults to False.
            p_flip (float): Probability of applying a random flip. Defaults to 0.1.
            p_rot (float): Probability of applying a random rotation. Defaults to 0.1.
            normalization (str): One of 'source_zscore', 'patch_zscore', or 'uint8'.

        Raises:
            ValueError: If split is not one of 'train', 'val', or 'test'.
        """
        
        self.h5_path = h5_path
        self.split = split
        self.return_metadata = return_metadata
        self.return_masks = return_masks
        valid_normalizations = {"source_zscore", "patch_zscore", "uint8"}
        if normalization not in valid_normalizations:
            raise ValueError(f"normalization must be one of {tuple(sorted(valid_normalizations))}, got {normalization!r}")
        self.normalization = normalization

        if self.split=='train' and augment:
            self.augment = True
        else:
            self.augment = False

        self.p_flip = p_flip
        self.p_rot = p_rot
 

        self.h5_file = None
        self.source_mean = None
        self.source_std = None
        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split not in split_map: 
            raise ValueError(f"split must be one of {tuple(split_map)}, got {split!r}")
        split_idx = split_map[split]

        with h5py.File(self.h5_path, 'r') as f:
            if self.normalization == "source_zscore":
                if "source_mean" not in f or "source_std" not in f:
                    raise ValueError("normalization='source_zscore' requires H5 datasets 'source_mean' and 'source_std'. Regenerate the H5 file with the current preprocessing pipeline.")
                self.source_mean = f["source_mean"][:].astype(np.float32)
                self.source_std = f["source_std"][:].astype(np.float32)
            source_splits = f['source_split'][:]
            patch_source_indices = f['source_index'][:]
            valid_mask = source_splits[patch_source_indices] == split_idx
            if source_index is not None:
                valid_mask &= (patch_source_indices == source_index)
            self.valid_indices = np.where(valid_mask)[0]
            self.patch_has_trail = f["patch_has_trail"][:][self.valid_indices].astype(bool)
            self.source_indices = patch_source_indices[self.valid_indices]
            self.patch_y0 = f["patch_y0"][:][self.valid_indices]
            self.patch_x0 = f["patch_x0"][:][self.valid_indices]

        self.pos_indices = np.where(self.patch_has_trail == True)[0]
        self.neg_indices = np.where(self.patch_has_trail == False)[0]

    def _normalize_image(self, image, source_index):
        if self.normalization == "source_zscore":
            eps = 1e-6
            source_std = self.source_std[source_index]
            if source_std < eps:
                source_std = 1.0
            return (image - self.source_mean[source_index]) / source_std
        if self.normalization == "patch_zscore":
            patch_mean = image.mean()
            patch_std = image.std()
            eps = 1e-6
            if patch_std < eps:
                patch_std = 1.0
            return (image - patch_mean) / patch_std
        return image / 255.0


    def __len__(self):
        """Returns the number of patches in the split."""

        return len(self.valid_indices)
    
    
    def __getitem__(self, idx):
        """
        Returns data for a single patch.

        Images are normalized according to ``normalization`` and returned as float tensors with a channel dimension added. Masks are binarized and returned as float tensors with a channel dimension added.

        Args:
            idx (int): Index into the split-filtered patch list

        Returns:
            x_tensor (torch.Tensor): Image patch of shape (1, patch_dim, patch_dim)
            y_tensor (torch.Tensor): Mask patch of shape (1, patch_dim, patch_dim),
                only if return_masks is True
            metadata (dict): Patch metadata, only if return_metadata is True.
                Keys: h5_index, source_index, patch_has_trail, patch_y0, patch_x0
        """

        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.images = self.h5_file['images']
            self.masks = self.h5_file['masks']
            
        real_idx = self.valid_indices[idx]
        patch_source_index = int(self.source_indices[idx])

        image = self.images[real_idx].astype(np.float32)
        need_mask = self.return_masks or (self.augment and self.patch_has_trail[idx])
        mask = np.ascontiguousarray(self.masks[real_idx] > 0).astype(np.float32) if need_mask else None

        if self.augment and self.patch_has_trail[idx]:
            # You can adjust these probabilities as needed
            image, mask = augment_image(image, mask, p_flip=self.p_flip, p_rot=self.p_rot)
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)
        
        image = self._normalize_image(image, patch_source_index)

        x_tensor = torch.from_numpy(image).float().unsqueeze(0)
        y_tensor = torch.from_numpy(mask).float().unsqueeze(0) if self.return_masks else None

        metadata = {"h5_index": int(real_idx), "source_index": patch_source_index, "patch_has_trail": bool(self.patch_has_trail[idx]), "patch_y0": int(self.patch_y0[idx]), "patch_x0": int(self.patch_x0[idx])}

        if self.return_metadata and self.return_masks:
            return x_tensor, y_tensor, metadata
        if self.return_metadata:
            return x_tensor, metadata
        if self.return_masks:
            return x_tensor, y_tensor
        return x_tensor
    

    def __del__(self): 
        """Closes the HDF5 file handle on dataset teardown."""
            
        if getattr(self, "h5_file", None) is not None:
            self.h5_file.close()

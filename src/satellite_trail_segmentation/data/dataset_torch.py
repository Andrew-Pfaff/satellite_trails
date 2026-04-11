import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5PatchDataset(Dataset):
    def __init__(self, h5_path, split='train'):
        self.h5_path = h5_path
        self.split = split

        self.h5_file = None
        split_map = {'train': 0, 'val': 1, 'test': 2}
        split_idx = split_map[split]

        with h5py.File(self.h5_path, 'r') as f:
            # 1. Get the split assignment (0, 1, or 2) for each original image
            source_splits = f['source_split'][:]
            
            # 2. Get the source image index for every single patch
            patch_source_indices = f['source_index'][:]
            
            # 3. Find which patches belong to the requested split
            # This maps the patch back to its parent image's split
            valid_mask = source_splits[patch_source_indices] == split_idx
            
            # Save the actual HDF5 indices we are allowed to use
            self.valid_indices = np.where(valid_mask)[0]


    def __len__(self):
        return len(self.valid_indices)
    
    
    def __getitem__(self, idx):
        # Open the file lazily for multiprocessing safety
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.images = self.h5_file['images']
            self.masks = self.h5_file['masks']
            
        # Get the true HDF5 row index for this sample
        real_idx = self.valid_indices[idx]
        
        # Read the 2D arrays from disk
        image = self.images[real_idx]
        mask = self.masks[real_idx]
        
        # Convert to PyTorch tensors
        # IMPORTANT: PyTorch CNNs expect a channel dimension (e.g., [1, 528, 528])
        # Since your patches are 2D, we use unsqueeze(0) to add a channel dimension of 1
        x_tensor = torch.from_numpy(image).float().unsqueeze(0)
        
        # For Binary Segmentation (BCEWithLogitsLoss), the mask also needs a channel dim and float type
        y_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        # Note: If using multi-class (CrossEntropyLoss), use this instead:
        # y_tensor = torch.from_numpy(mask).long() 
        
        return x_tensor, y_tensor
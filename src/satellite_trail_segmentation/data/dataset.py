import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5PatchDataset(Dataset):
    def __init__(self, h5_path, split='train', return_metadata=False, source_index=None):
        self.h5_path = h5_path
        self.split = split
        self.return_metadata = return_metadata

        self.h5_file = None
        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split not in split_map: 
            raise ValueError(f"split must be one of {tuple(split_map)}, got {split!r}")
        split_idx = split_map[split]

        with h5py.File(self.h5_path, 'r') as f:
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


    def __len__(self):
        return len(self.valid_indices)
    
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.images = self.h5_file['images']
            self.masks = self.h5_file['masks']
            
        real_idx = self.valid_indices[idx]
        
        image = self.images[real_idx].astype(np.float32) / 255.0
        mask = (self.masks[real_idx] > 0).astype(np.float32)
        
        x_tensor = torch.from_numpy(image).float().unsqueeze(0)
        y_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        metadata = {"h5_index": int(real_idx), "source_index": int(self.source_indices[idx]), "patch_has_trail": bool(self.patch_has_trail[idx]), "patch_y0": int(self.patch_y0[idx]), "patch_x0": int(self.patch_x0[idx])}

        return (x_tensor, y_tensor, metadata) if self.return_metadata else (x_tensor, y_tensor)      
    

    def __del__(self): 
        self.h5_file.close() if getattr(self, "h5_file", None) is not None else None

# tests/conftest.py
import numpy as np
import h5py
from PIL import Image

import pytest

@pytest.fixture
def sample_patch():
    """Patch with a known horizontal trail for testing mask preservation."""
    image = np.random.randint(0, 256, (528, 528), dtype=np.uint8)
    mask = np.zeros((528, 528), dtype=np.uint8)
    image[100, :] = 255
    mask[100, :] = 1
    return image, mask

@pytest.fixture
def dummy_image_dir(tmp_path):
    """Creates a temporary directory with dummy image and mask pairs."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Create 3 valid pairs
    for i in range(3):
        img_path = img_dir / f"img_{i}.fits_full.png"
        mask_path = img_dir / f"img_{i}_mask.png"
        
        # Save empty 1056x1056 images (exactly divisible by 528)
        img_array = np.zeros((1056, 1056), dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)
        Image.fromarray(img_array).save(mask_path)
        
    return img_dir


@pytest.fixture
def dummy_h5_file(tmp_path):
    """Creates a dummy h5 file structured exactly like the output of preprocessing."""
    h5_path = tmp_path / "dummy_dataset.h5"
    
    total_patches = 10
    patch_dim = 64
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset("images", data=np.random.randint(0, 255, (total_patches, patch_dim, patch_dim), dtype=np.uint8))
        f.create_dataset("masks", data=np.random.randint(0, 2, (total_patches, patch_dim, patch_dim), dtype=np.uint8))
        
        # 5 train patches (split=0), 3 val patches (split=1), 2 test patches (split=2)
        f.create_dataset("source_split", data=np.array([0, 0, 1, 0, 1, 2, 0, 2, 0, 1], dtype=np.uint8))
        
        # Assume 1 patch per source for simplicity
        f.create_dataset("source_index", data=np.arange(total_patches, dtype=np.int32))
        
        # Let some patches have trails
        f.create_dataset("patch_has_trail", data=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8))
        
        f.create_dataset("patch_y0", data=np.zeros(total_patches, dtype=np.int32))
        f.create_dataset("patch_x0", data=np.zeros(total_patches, dtype=np.int32))
        
    return str(h5_path)
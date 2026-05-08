import torch
import pytest

from satellite_trail_segmentation.data.dataset import H5PatchDataset


def test_dataset_initialization(dummy_h5_file):
    # Test valid splits lengths
    train_dataset = H5PatchDataset(dummy_h5_file, split='train')
    assert len(train_dataset) == 5
    
    val_dataset = H5PatchDataset(dummy_h5_file, split='val')
    assert len(val_dataset) == 3

    test_dataset = H5PatchDataset(dummy_h5_file, split='test')
    assert len(test_dataset) == 2

def test_dataset_invalid_split(dummy_h5_file):
    with pytest.raises(ValueError, match="split must be one of"):
        H5PatchDataset(dummy_h5_file, split='invalid_split')


def test_dataset_source_index_filtering(dummy_h5_file):
    # Only pick patches matching a specific source index (e.g., index 3 which is train)
    dataset = H5PatchDataset(dummy_h5_file, split='train', source_index=3)
    assert len(dataset) == 1
    assert dataset.valid_indices[0] == 3


def test_dataset_getitem(dummy_h5_file):
    dataset = H5PatchDataset(dummy_h5_file, split='train', return_metadata=False)
    
    x, y = dataset[0]
    
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    # Checking shape: (Channels, Height, Width) -> Channel should be 1
    assert x.shape == (1, 64, 64) 
    assert y.shape == (1, 64, 64)
    # Check normalization
    assert torch.max(x) <= 1.0 
    assert torch.min(x) >= 0.0

def test_dataset_getitem_metadata(dummy_h5_file):
    dataset = H5PatchDataset(dummy_h5_file, split='train', return_metadata=True)
    
    x, y, metadata = dataset[0]
    
    assert isinstance(metadata, dict)
    assert "h5_index" in metadata
    assert "patch_has_trail" in metadata
    assert "source_index" in metadata
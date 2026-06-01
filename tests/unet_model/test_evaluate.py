import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from satellite_trail_segmentation.unet_model.unet_evaluate import evaluate_patches, image_threshold

def test_image_threshold():
    """Test numpy binarization arrays."""
    image = np.array([[0.1, 0.6], [0.4, 0.9]])
    expected_output = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    
    result = image_threshold(image, threshold=0.5)
    np.testing.assert_array_equal(result, expected_output)

@patch("satellite_trail_segmentation.model.evaluate.H5PatchDataset")
@patch("satellite_trail_segmentation.model.evaluate.DataLoader")
def test_evaluate_patches(MockDataLoader, MockDataset, dummy_unet, dummy_image, dummy_mask):
    """Test evaluation over a dummy dataloader loop."""
    
    dummy_metadata = {"patch_y0": torch.tensor([0]), "patch_x0": torch.tensor([0])}
    
    mock_loader = MagicMock()
    mock_loader.__len__.return_value = 1
    mock_loader.__iter__.return_value = iter([(dummy_image, dummy_mask, dummy_metadata)])
    MockDataLoader.return_value = mock_loader

    test_loss = evaluate_patches(dummy_unet, "dummy/path.h5", "test", batch_size=2)
    
    assert isinstance(test_loss, float)
    assert test_loss >= 0.0
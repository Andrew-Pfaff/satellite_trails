import pytest
import torch
import numpy as np
from satellite_trail_segmentation.unet_model.unet import UNet

@pytest.fixture
def dummy_image():
    return torch.randn(2, 1, 64, 64)

@pytest.fixture
def dummy_mask():
    return torch.randint(0, 2, (2, 1, 64, 64)).float()

@pytest.fixture
def dummy_unet():
    return UNet(in_channels=1, out_channels=1, kernel_size=3, base_channels=4, dropout=0.0)

@pytest.fixture
def dummy_dataset_batch(dummy_image, dummy_mask):
    return [(dummy_image, dummy_mask)]
import numpy as np
import pytest
import torch

from satellite_trail_segmentation.unet_model.unet import UNet


@pytest.fixture(autouse=True)
def seed_all():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def dummy_image():
    return torch.randn(2, 1, 32, 32)


@pytest.fixture
def dummy_mask():
    return torch.randint(0, 2, (2, 1, 32, 32)).float()


@pytest.fixture
def dummy_unet():
    return UNet(in_channels=1, out_channels=1, kernel_size=3, base_channels=4, dropout=0.0)


class TinySegDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = torch.stack([torch.zeros(1, 32, 32), torch.ones(1, 32, 32)])
        self.masks = torch.stack([torch.zeros(1, 32, 32), torch.ones(1, 32, 32)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


@pytest.fixture
def dummy_dataset_batch(dummy_image, dummy_mask):
    return [(dummy_image, dummy_mask)]


@pytest.fixture
def tiny_seg_dataset():
    return TinySegDataset()


@pytest.fixture
def tiny_unet_model():
    return UNet(in_channels=1, out_channels=1, kernel_size=3, base_channels=4, dropout=0.0)

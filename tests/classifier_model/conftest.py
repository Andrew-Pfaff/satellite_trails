import numpy as np
import pytest
import torch

from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier


@pytest.fixture(autouse=True)
def seed_all_classifier():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def tiny_classifier():
    return TrailClassifier(in_channels=1, kernel_size=3, base_channels=4, dropout=0.2)


class TinyClassifierDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        image = torch.full((1, 32, 32), float(idx))
        metadata = {"patch_has_trail": torch.tensor(idx % 2, dtype=torch.float32)}
        return image, metadata


@pytest.fixture
def tiny_classifier_dataset():
    return TinyClassifierDataset()

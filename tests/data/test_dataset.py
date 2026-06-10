import gc

import numpy as np
import pytest
import torch

from satellite_trail_segmentation.data.dataset import H5PatchDataset


def test_dataset_split_lengths_and_filtering(dummy_h5_file):
    assert len(H5PatchDataset(dummy_h5_file, split="train")) == 4
    assert len(H5PatchDataset(dummy_h5_file, split="val")) == 4
    assert len(H5PatchDataset(dummy_h5_file, split="test")) == 4

    dataset = H5PatchDataset(dummy_h5_file, split="train", source_index=0)
    assert len(dataset) == 4
    assert dataset.valid_indices.tolist() == [0, 1, 2, 3]
    assert len(H5PatchDataset(dummy_h5_file, split="train", source_index=1)) == 0
    assert dataset.pos_indices.tolist() == [1, 2, 3]
    assert dataset.neg_indices.tolist() == [0]

    assert len(H5PatchDataset(dummy_h5_file, split="val", source_index=1)) == 4
    assert len(H5PatchDataset(dummy_h5_file, split="test", source_index=2)) == 4


def test_dataset_getitem_normalizes_and_returns_metadata(dummy_h5_file):
    dataset = H5PatchDataset(dummy_h5_file, split="train", return_metadata=True)
    x, y, metadata = dataset[1]

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.shape == (1, 4, 4)
    assert y.shape == (1, 4, 4)
    assert torch.all((x >= 0) & (x <= 1))
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}
    assert metadata == {"h5_index": 1, "source_index": 0, "patch_has_trail": True, "patch_y0": 0, "patch_x0": 4}


def test_dataset_without_masks_and_with_metadata(dummy_h5_file):
    dataset = H5PatchDataset(dummy_h5_file, split="train", return_metadata=True, return_masks=False)
    x, metadata = dataset[0]
    assert x.shape == (1, 4, 4)
    assert metadata["h5_index"] == 0
    assert "patch_has_trail" in metadata

    x_only = H5PatchDataset(dummy_h5_file, split="train", return_metadata=False, return_masks=False)[0]
    assert isinstance(x_only, torch.Tensor)
    assert x_only.shape == (1, 4, 4)


def test_dataset_val_and_test_metadata_rows(dummy_h5_file):
    val_dataset = H5PatchDataset(dummy_h5_file, split="val", return_metadata=True, return_masks=False)
    test_dataset = H5PatchDataset(dummy_h5_file, split="test", return_metadata=True, return_masks=False)

    _, val_first = val_dataset[0]
    _, val_last = val_dataset[3]
    _, test_first = test_dataset[0]
    _, test_last = test_dataset[3]

    assert val_first == {"h5_index": 4, "source_index": 1, "patch_has_trail": False, "patch_y0": 0, "patch_x0": 0}
    assert val_last == {"h5_index": 7, "source_index": 1, "patch_has_trail": True, "patch_y0": 4, "patch_x0": 4}
    assert test_first == {"h5_index": 8, "source_index": 2, "patch_has_trail": True, "patch_y0": 0, "patch_x0": 0}
    assert test_last == {"h5_index": 11, "source_index": 2, "patch_has_trail": False, "patch_y0": 4, "patch_x0": 4}


def test_dataset_zscore_standardization(dummy_h5_file):
    dataset = H5PatchDataset(dummy_h5_file, split="train", zscore_standardization=True)
    x, _ = dataset[2]
    assert abs(float(x.mean())) < 1e-6
    assert pytest.approx(float(x.std()), rel=0.2) == 1.0

    const_dataset = H5PatchDataset(dummy_h5_file, split="train", zscore_standardization=True)
    x0, _ = const_dataset[1]
    assert torch.allclose(x0, torch.zeros_like(x0), atol=1e-6)


def test_dataset_train_augmentation_only_for_positive_patches(monkeypatch, dummy_h5_file):
    calls = []

    def fake_augment(image, mask, **kwargs):
        calls.append((image.copy(), mask.copy()))
        return image + 1, mask

    monkeypatch.setattr("satellite_trail_segmentation.data.dataset.augment_image", fake_augment)

    dataset = H5PatchDataset(dummy_h5_file, split="train", augment=True)
    x_neg, y_neg = dataset[0]
    x_pos, y_pos = dataset[1]
    val_dataset = H5PatchDataset(dummy_h5_file, split="val", augment=True)
    test_dataset = H5PatchDataset(dummy_h5_file, split="test", augment=True)
    _ = val_dataset[1]
    _ = test_dataset[0]

    assert len(calls) == 1
    assert torch.allclose(x_neg, torch.zeros_like(x_neg))
    assert y_neg.sum() == 0
    assert y_pos.sum() > 0
    assert torch.max(x_pos) <= 1.0
    assert set(torch.unique(y_pos).tolist()) <= {0.0, 1.0}


def test_dataset_lazy_open_and_close(dummy_h5_file):
    dataset = H5PatchDataset(dummy_h5_file, split="train")
    assert dataset.h5_file is None
    _ = dataset[0]
    assert dataset.h5_file is not None
    file_ref = dataset.h5_file
    del dataset
    gc.collect()
    assert not file_ref.id.valid

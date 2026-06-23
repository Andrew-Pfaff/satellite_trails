import gc

import h5py
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
    dataset = H5PatchDataset(dummy_h5_file, split="train", return_metadata=True, normalization="uint8")
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


def test_dataset_normalization_modes(dummy_h5_file):
    source_dataset = H5PatchDataset(dummy_h5_file, split="train", normalization="source_zscore")
    x_source, _ = source_dataset[2]
    image = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 0, 0], [4, 4, 4, 4]], dtype=np.float32)
    expected_source = (image - 5.0) / 5.0
    assert torch.allclose(x_source.squeeze(0), torch.from_numpy(expected_source))

    patch_dataset = H5PatchDataset(dummy_h5_file, split="train", normalization="patch_zscore")
    x_patch, _ = patch_dataset[2]
    assert abs(float(x_patch.mean())) < 1e-6
    assert pytest.approx(float(x_patch.std(unbiased=False)), rel=0.2) == 1.0

    const_patch_dataset = H5PatchDataset(dummy_h5_file, split="train", normalization="patch_zscore")
    x_const, _ = const_patch_dataset[1]
    assert torch.allclose(x_const, torch.zeros_like(x_const), atol=1e-6)

    uint8_dataset = H5PatchDataset(dummy_h5_file, split="train", normalization="uint8")
    x_uint8, _ = uint8_dataset[1]
    assert torch.allclose(x_uint8, torch.full_like(x_uint8, 5.0 / 255.0))


def test_source_zscore_requires_source_stats(tmp_path):
    h5_path = tmp_path / "missing_source_stats.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("images", data=np.zeros((1, 4, 4), dtype=np.uint8))
        f.create_dataset("masks", data=np.zeros((1, 4, 4), dtype=np.uint8))
        f.create_dataset("source_split", data=np.array([0], dtype=np.uint8))
        f.create_dataset("source_index", data=np.array([0], dtype=np.int32))
        f.create_dataset("patch_has_trail", data=np.array([0], dtype=np.uint8))
        f.create_dataset("patch_y0", data=np.array([0], dtype=np.int32))
        f.create_dataset("patch_x0", data=np.array([0], dtype=np.int32))

    with pytest.raises(ValueError, match="source_zscore.*source_mean.*source_std"):
        H5PatchDataset(str(h5_path), split="train", normalization="source_zscore")


def test_dataset_train_augmentation_only_for_positive_patches(monkeypatch, dummy_h5_file):
    calls = []

    def fake_augment(image, mask, **kwargs):
        calls.append((image.copy(), mask.copy()))
        return image + 1, mask

    monkeypatch.setattr("satellite_trail_segmentation.data.dataset.augment_image", fake_augment)

    dataset = H5PatchDataset(dummy_h5_file, split="train", augment=True, normalization="uint8")
    x_neg, y_neg = dataset[0]
    x_pos, y_pos = dataset[1]
    val_dataset = H5PatchDataset(dummy_h5_file, split="val", augment=True, normalization="uint8")
    test_dataset = H5PatchDataset(dummy_h5_file, split="test", augment=True, normalization="uint8")
    _ = val_dataset[1]
    _ = test_dataset[0]

    assert len(calls) == 1
    assert torch.allclose(x_neg, torch.zeros_like(x_neg))
    assert y_neg.sum() == 0
    assert y_pos.sum() > 0
    assert torch.allclose(x_pos, torch.full_like(x_pos, 1.0 + 5.0 / 255.0))
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

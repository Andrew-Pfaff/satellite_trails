import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from satellite_trail_segmentation.evaluation.unet_evaluate import evaluate_dataset_unet, image_threshold, recreate_full_field_pred


def test_image_threshold_strict_greater():
    image = np.array([[0.5, 0.51]], dtype=np.float32)
    out = image_threshold(image, threshold=0.5)
    np.testing.assert_array_equal(out, np.array([[0, 255]], dtype=np.uint8))


class ConstantUNet(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(float(logit)))

    def forward(self, x):
        return torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype) * self.logit


class EchoUNet(torch.nn.Module):
    def forward(self, x):
        return x


def _reconstruct_source_array(h5_path, source_index, dataset_name, transform, patch_dim=4):
    with h5py.File(h5_path, "r") as f:
        full = np.zeros(tuple(f.attrs["full_shape"]), dtype=np.float32)
        rows = np.flatnonzero(f["source_index"][:] == source_index)
        for row in rows:
            y0 = int(f["patch_y0"][row])
            x0 = int(f["patch_x0"][row])
            full[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = transform(f[dataset_name][row])
    return full


def test_recreate_full_field_pred(dummy_h5_file):
    model = EchoUNet()
    full_image, full_pred, full_mask = recreate_full_field_pred(model, dummy_h5_file, "train", 0, batch_size=2, patch_dim=4)

    expected_image = _reconstruct_source_array(dummy_h5_file, 0, "images", lambda patch: patch.astype(np.float32) / 255.0)
    expected_mask = _reconstruct_source_array(dummy_h5_file, 0, "masks", lambda patch: (patch > 0).astype(np.float32))
    expected_pred = 1 / (1 + np.exp(-expected_image))

    np.testing.assert_allclose(full_image, expected_image, rtol=0, atol=1e-7)
    np.testing.assert_allclose(full_pred, expected_pred, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(full_mask, expected_mask)


def test_evaluate_dataset_unet(monkeypatch, dummy_h5_file):
    model = ConstantUNet(0.0)
    monkeypatch.setattr(
        "satellite_trail_segmentation.evaluation.unet_evaluate.DataLoader",
        lambda dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True: TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0),
    )
    metrics, fpr, tpr, thresholds, optimal_threshold, roc_auc = evaluate_dataset_unet(model, dummy_h5_file, "train", pred_thresholds=[0.2, 0.8], batch_size=2)
    assert set(metrics) == {0.2, 0.8}
    assert {key: metrics[0.2][key] for key in ("tp", "fp", "fn", "tn")} == {"tp": 24.0, "fp": 40.0, "fn": 0.0, "tn": 0.0}
    assert {key: metrics[0.8][key] for key in ("tp", "fp", "fn", "tn")} == {"tp": 0.0, "fp": 0.0, "fn": 24.0, "tn": 40.0}
    assert fpr.ndim == tpr.ndim == thresholds.ndim == 1
    assert fpr.shape == tpr.shape == thresholds.shape
    assert 0.0 <= roc_auc <= 1.0
    assert optimal_threshold is not None

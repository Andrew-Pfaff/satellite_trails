import h5py
import numpy as np
import torch

from satellite_trail_segmentation.evaluation.classifier_evaluate import _predict, evaluate_dataset_classifier, recreate_full_field


class ConstantClassifier(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(float(logit)))

    def forward(self, x):
        return torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype) * self.logit


class MeanThresholdClassifier(torch.nn.Module):
    def forward(self, x):
        patch_mean = x.mean(dim=(1, 2, 3), keepdim=False).view(-1, 1)
        return torch.where(patch_mean > 0.015, torch.full_like(patch_mean, 10.0), torch.full_like(patch_mean, -10.0))


def _reconstruct_source_array(h5_path, source_index, fill_from_row, patch_dim=4):
    with h5py.File(h5_path, "r") as f:
        full = np.zeros(tuple(f.attrs["full_shape"]), dtype=np.float32)
        rows = np.flatnonzero(f["source_index"][:] == source_index)
        for row in rows:
            y0 = int(f["patch_y0"][row])
            x0 = int(f["patch_x0"][row])
            full[y0 : y0 + patch_dim, x0 : x0 + patch_dim] = fill_from_row(f, row)
    return full


def test_predict_threshold_behavior():
    logits = torch.tensor([[-10.0], [0.0], [10.0]])
    assert _predict(logits, threshold=0.5).tolist() == [[0], [1], [1]]


def test_recreate_full_field(dummy_h5_file):
    model = MeanThresholdClassifier()
    full_image, full_pred, full_mask, full_overlay = recreate_full_field(model, dummy_h5_file, "train", 0, batch_size=2, patch_dim=4, threshold=0.5)

    expected_image = _reconstruct_source_array(dummy_h5_file, 0, lambda f, row: f["images"][row].astype(np.float32) / 255.0)
    expected_pred = _reconstruct_source_array(
        dummy_h5_file,
        0,
        lambda f, row: float((f["images"][row].astype(np.float32) / 255.0).mean() > 0.015),
    )
    expected_mask = _reconstruct_source_array(dummy_h5_file, 0, lambda f, row: float(f["patch_has_trail"][row]))

    np.testing.assert_allclose(full_image, expected_image, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(full_pred, expected_pred)
    np.testing.assert_array_equal(full_mask, expected_mask)
    assert full_overlay.shape == (8, 8, 3)


def test_evaluate_dataset_classifier(dummy_h5_file):
    model = ConstantClassifier(0.0)
    metrics, image_wise_counts = evaluate_dataset_classifier(model, dummy_h5_file, "train", pred_thresholds=[0.2, 0.8], batch_size=2)
    assert set(metrics) == {0.2, 0.8}
    assert set(image_wise_counts) == {0.2, 0.8}
    assert {key: metrics[0.2][key] for key in ("tp", "fp", "fn", "tn")} == {"tp": 3.0, "fp": 1.0, "fn": 0.0, "tn": 0.0}
    assert {key: metrics[0.8][key] for key in ("tp", "fp", "fn", "tn")} == {"tp": 0.0, "fp": 0.0, "fn": 3.0, "tn": 1.0}
    assert image_wise_counts[0.2] == {0: {"tp": 3.0, "fp": 1.0, "fn": 0.0, "tn": 0.0}}
    assert image_wise_counts[0.8] == {0: {"tp": 0.0, "fp": 0.0, "fn": 3.0, "tn": 1.0}}

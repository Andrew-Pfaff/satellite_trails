import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import satellite_trail_segmentation.utils.visualizations as visualizations
from satellite_trail_segmentation.utils.visualizations import (
    plot_full_field,
    plot_loss_curves,
    plot_prediction_mask,
    plot_pred_residual,
    plot_roc_curve,
    plot_segmentation_postprocess_comparison,
    plot_threshold_metrics,
)


def test_plot_loss_curves_creates_file(tmp_path):
    path = tmp_path / "loss.png"
    plot_loss_curves([1.0, 0.5], [1.2, 0.6], path)
    assert path.exists()
    assert plt.get_fignums() == []


def test_plot_full_field_thresholds_and_creates_file(tmp_path):
    path = tmp_path / "full_field.png"
    full_image = np.zeros((8, 8), dtype=np.float32)
    full_pred = np.array([[0.1, 0.7] * 4] * 8, dtype=np.float32)
    full_mask = np.ones((8, 8), dtype=np.float32)
    plot_full_field(full_image, full_pred, full_mask, save_path=path, threshold=0.5)
    assert path.exists()
    assert plt.get_fignums() == []


def test_plot_segmentation_postprocess_comparison_creates_file(tmp_path):
    path = tmp_path / "postprocess_comparison.png"
    image = np.zeros((8, 8), dtype=np.float32)
    mask = np.ones((8, 8), dtype=np.uint8)
    prediction = np.eye(8, dtype=np.uint8)
    postprocessed = np.fliplr(prediction)

    plot_segmentation_postprocess_comparison(image, mask, prediction, postprocessed, save_path=path)

    assert path.exists()
    assert plt.get_fignums() == []


def test_plot_segmentation_postprocess_comparison_without_mask_creates_file(tmp_path):
    path = tmp_path / "postprocess_comparison_no_mask.png"
    image = np.zeros((8, 8), dtype=np.float32)
    prediction = np.eye(8, dtype=np.float32)
    postprocessed = np.fliplr(prediction)

    plot_segmentation_postprocess_comparison(image, None, prediction, postprocessed, save_path=path, threshold=0.5)

    assert path.exists()
    assert plt.get_fignums() == []


def test_plot_segmentation_postprocess_comparison_accepts_multiple_masks(tmp_path):
    path = tmp_path / "postprocess_comparison_multiple.png"
    image = np.zeros((8, 8), dtype=np.float32)
    mask = np.ones((8, 8), dtype=np.uint8)
    prediction = np.eye(8, dtype=np.uint8)
    postprocessed = {"ASTA-only": prediction, "ASTA gap-fill": np.fliplr(prediction)}

    plot_segmentation_postprocess_comparison(image, mask, prediction, postprocessed, save_path=path, include_image=False, error_overlay=False)

    assert path.exists()
    assert plt.get_fignums() == []


def test_plot_prediction_mask_creates_file_with_and_without_mask(tmp_path):
    image = np.zeros((8, 8), dtype=np.float32)
    prediction = np.eye(8, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)

    with_mask_path = tmp_path / "prediction_mask_with_mask.png"
    without_mask_path = tmp_path / "prediction_mask_without_mask.png"

    plot_prediction_mask(image, prediction, mask=mask, save_path=with_mask_path)
    plot_prediction_mask(image, prediction, save_path=without_mask_path)

    assert with_mask_path.exists()
    assert without_mask_path.exists()
    assert plt.get_fignums() == []


def test_error_color_image_marks_false_negative_and_false_positive():
    prediction = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    color_image = visualizations._error_color_image(prediction, mask)

    np.testing.assert_array_equal(color_image[0, 0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(color_image[0, 1], np.array([0.0, 1.0, 0.0]))


def test_plot_roc_curve_creates_file(tmp_path):
    path = tmp_path / "roc.png"
    fpr = np.array([0.0, 0.5, 1.0])
    tpr = np.array([0.0, 0.8, 1.0])
    thresholds = np.array([0.9, 0.5, 0.1])
    plot_roc_curve(fpr, tpr, thresholds, 0.83, 0.5, path)
    assert path.exists()
    assert plt.get_fignums() == []


def test_plot_pred_residual_creates_file(tmp_path):
    path = tmp_path / "residual.png"
    full_pred = np.full((8, 8), 0.7, dtype=np.float32)
    full_mask = np.zeros((8, 8), dtype=np.float32)
    plot_pred_residual(full_pred, full_mask, path)
    assert path.exists()
    assert plt.get_fignums() == []


def test_plot_threshold_metrics_creates_file(tmp_path):
    path = tmp_path / "thresholds.png"
    metrics = {
        0.2: {"iou": 0.1, "precision": 0.2, "recall": 0.3, "dice": 0.4},
        0.8: {"iou": 0.5, "precision": 0.6, "recall": 0.7, "dice": 0.8},
    }
    plot_threshold_metrics(metrics, save_path=path)
    assert path.exists()
    assert plt.get_fignums() == []

import numpy as np
import cv2
import pytest
import torch

from satellite_trail_segmentation.postprocess.hough import (
    binarize_prediction,
    hough_gap_fill,
    morphological_close,
    postprocess_segmentation,
    remove_small_components,
    standardize_binary_mask,
    to_numpy_2d,
)


def test_to_numpy_2d_accepts_numpy_and_does_not_mutate():
    pred = np.array([[[0.1, 0.9], [0.2, 0.8]]], dtype=np.float32)
    original = pred.copy()

    out = to_numpy_2d(pred)

    assert out.shape == (2, 2)
    np.testing.assert_array_equal(pred, original)


def test_to_numpy_2d_accepts_torch_tensor():
    pred = torch.tensor([[[0.1, 0.9], [0.2, 0.8]]])

    out = to_numpy_2d(pred)

    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32))


def test_to_numpy_2d_rejects_ambiguous_shape():
    with pytest.raises(ValueError):
        to_numpy_2d(np.zeros((2, 3, 4), dtype=np.float32))


def test_binarize_prediction_thresholds_probability_map():
    pred = np.array([[0.57, 0.58, 0.59]], dtype=np.float32)

    out = binarize_prediction(pred, threshold=0.58)

    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, np.array([[0, 0, 255]], dtype=np.uint8))


def test_binarize_prediction_thresholds_two_value_probability_map():
    pred = np.array([[0.2, 0.8]], dtype=np.float32)

    out = binarize_prediction(pred, threshold=0.58)

    np.testing.assert_array_equal(out, np.array([[0, 255]], dtype=np.uint8))


def test_binarize_prediction_standardizes_binary_like_mask():
    pred = np.array([[0, 1, 255]], dtype=np.uint8)

    out = binarize_prediction(pred)

    np.testing.assert_array_equal(out, np.array([[0, 255, 255]], dtype=np.uint8))


def test_standardize_binary_mask_treats_nonzero_as_foreground():
    mask = np.array([[0, 1, 255]], dtype=np.uint8)

    out = standardize_binary_mask(mask)

    np.testing.assert_array_equal(out, np.array([[0, 255, 255]], dtype=np.uint8))


def test_hough_gap_fill_draws_with_supplied_thickness(monkeypatch):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 1:4] = 255
    mask[5, 6:9] = 255
    original = mask.copy()
    drawn = {}

    def fake_lines(*args, **kwargs):
        return np.array([[[1, 5, 8, 5]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(img, pt1, pt2, color, thickness=1):
        drawn["thickness"] = thickness
        return original_line(img, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    out = hough_gap_fill(mask, thickness=5, hough_threshold=1, min_line_length=1, max_line_gap=5)

    np.testing.assert_array_equal(mask, original)
    assert drawn["thickness"] == 5
    assert out.dtype == np.uint8
    assert out[5, 4] == 255


def test_morphological_close_fills_small_gap():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[3, 1:3] = 255
    mask[3, 4:6] = 255

    out = morphological_close(mask, kernel_size=3)

    assert out[3, 3] == 255


def test_remove_small_components_removes_tiny_objects():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1, 1] = 255
    mask[10:15, 10:15] = 255

    out = remove_small_components(mask, min_size=10)

    assert out[1, 1] == 0
    assert out[12, 12] == 255


def test_postprocess_segmentation_returns_clean_binary_mask(monkeypatch):
    pred = np.zeros((40, 40), dtype=np.float32)
    pred[20:24, 5:15] = 1
    pred[20:24, 25:35] = 1
    pred[1, 1] = 1
    original = pred.copy()

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 21, 34, 21]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    out = postprocess_segmentation(
        pred,
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=3,
        min_component_size=20,
    )

    np.testing.assert_array_equal(pred, original)
    assert out.shape == pred.shape
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})
    assert out[21, 20] == 255
    assert out[1, 1] == 0

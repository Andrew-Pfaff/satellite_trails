import cv2
import numpy as np

from satellite_trail_segmentation.postprocess.adaptive_hough import (
    hough_gap_fill_contour_width,
    postprocess_segmentation_contour_width,
)


def test_hough_gap_fill_contour_width_draws_with_adaptive_thickness(monkeypatch):
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[13:18, 2:10] = 255
    mask[13:18, 20:28] = 255
    original = mask.copy()
    drawn = {}

    def fake_lines(*args, **kwargs):
        return np.array([[[2, 15, 27, 15]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(img, pt1, pt2, color, thickness=1):
        drawn["thickness"] = thickness
        return original_line(img, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    out = hough_gap_fill_contour_width(
        mask,
        hough_threshold=1,
        min_line_length=1,
        max_line_gap=20,
        max_thickness=10,
    )

    np.testing.assert_array_equal(mask, original)
    assert drawn["thickness"] > 1
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})
    assert out[15, 15] == 255


def test_hough_gap_fill_contour_width_uses_fallback_without_matching_contour(monkeypatch):
    mask = np.zeros((30, 30), dtype=np.uint8)
    drawn = {}

    def fake_lines(*args, **kwargs):
        return np.array([[[2, 15, 27, 15]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(img, pt1, pt2, color, thickness=1):
        drawn["thickness"] = thickness
        return original_line(img, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    out = hough_gap_fill_contour_width(
        mask,
        hough_threshold=1,
        min_line_length=1,
        max_line_gap=20,
        fallback_thickness=1,
    )

    assert drawn["thickness"] == 1
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_postprocess_segmentation_contour_width_returns_clean_binary_mask(monkeypatch):
    pred = np.zeros((40, 40), dtype=np.float32)
    pred[18:23, 5:15] = 1
    pred[18:23, 25:35] = 1
    pred[1, 1] = 1
    original = pred.copy()

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    out = postprocess_segmentation_contour_width(
        pred,
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=3,
        min_component_size=20,
        max_thickness=10,
    )

    np.testing.assert_array_equal(pred, original)
    assert out.shape == pred.shape
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})
    assert out[20, 20] == 255
    assert out[1, 1] == 0

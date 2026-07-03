import cv2
import numpy as np
import pytest

from satellite_trail_segmentation.postprocess.pipeline import (
    analyze_hough_lines,
    cluster_hough_lines,
    hough_gap_fill_contour_width,
    hough_gap_fill_fwhm_sampled,
    perpendicular_mask_widths,
    postprocess_segmentation,
    thickness_from_fwhm_samples,
)


def test_one_pixel_pipeline_draws_with_thickness_one(monkeypatch):
    pred = np.zeros((40, 40), dtype=np.float32)
    pred[20:24, 5:15] = 1
    pred[20:24, 25:35] = 1
    original = pred.copy()
    drawn = {}

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 21, 34, 21]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(img, pt1, pt2, color, thickness=1):
        drawn["thickness"] = thickness
        return original_line(img, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    out = postprocess_segmentation(
        pred,
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=3,
        min_component_size=20,
    )

    np.testing.assert_array_equal(pred, original)
    assert drawn["thickness"] == 1
    assert out.shape == pred.shape
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})
    assert out[21, 20] == 255


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


def test_duplicate_hough_lines_cluster_together():
    lines = np.array(
        [
            [[2, 15, 27, 15]],
            [[2, 16, 27, 16]],
            [[2, 14, 27, 14]],
            [[2, 5, 27, 5]],
        ],
        dtype=np.int32,
    )

    records = analyze_hough_lines(lines)
    clusters = cluster_hough_lines(records, angle_degrees=3, distance=3)

    cluster_sizes = sorted(len(cluster) for cluster in clusters)
    assert cluster_sizes == [1, 3]


def test_contour_width_pipeline_returns_clean_binary_mask(monkeypatch):
    pred = np.zeros((40, 40), dtype=np.float32)
    pred[18:23, 5:15] = 1
    pred[18:23, 25:35] = 1
    pred[1, 1] = 1
    original = pred.copy()

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    out = postprocess_segmentation(
        pred,
        gap_fill_mode="contour_width",
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


def test_perpendicular_mask_widths_samples_horizontal_line_width():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[13:18, 2:28] = 255

    widths = perpendicular_mask_widths(
        mask,
        np.array([2, 15, 27, 15], dtype=np.int32),
        num_samples=5,
        max_width_search=10,
    )

    assert widths == [5, 5, 5, 5, 5]


def test_thickness_from_fwhm_samples_uses_median_sampled_width():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[13:18, 2:14] = 255
    mask[13:18, 14:16] = 255
    mask[11:20, 16:28] = 255

    thickness = thickness_from_fwhm_samples(
        mask,
        np.array([2, 15, 27, 15], dtype=np.int32),
        min_thickness=1,
        max_thickness=10,
        width_samples=5,
        max_width_search=10,
    )

    assert thickness == 5


def test_thickness_from_fwhm_samples_is_uncapped_by_default():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[10:21, 2:28] = 255

    thickness = thickness_from_fwhm_samples(
        mask,
        np.array([2, 15, 27, 15], dtype=np.int32),
        min_thickness=1,
        width_samples=5,
        max_width_search=10,
    )

    assert thickness == 11


def test_hough_gap_fill_fwhm_sampled_draws_sampled_thickness(monkeypatch):
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[13:18, 2:10] = 255
    mask[13:18, 20:28] = 255
    drawn = {}

    def fake_lines(*args, **kwargs):
        return np.array([[[2, 15, 27, 15]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(img, pt1, pt2, color, thickness=1):
        drawn["thickness"] = thickness
        return original_line(img, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    out = hough_gap_fill_fwhm_sampled(
        mask,
        hough_threshold=1,
        min_line_length=1,
        max_line_gap=20,
        max_thickness=10,
        width_samples=5,
        max_width_search=10,
    )

    assert drawn["thickness"] == 5
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})
    assert out[15, 15] == 255


def test_fwhm_sampled_draws_one_centerline_for_duplicate_hough_lines(monkeypatch):
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[13:18, 2:10] = 255
    mask[13:18, 20:28] = 255
    drawn = []

    def fake_lines(*args, **kwargs):
        return np.array(
            [
                [[2, 15, 27, 15]],
                [[2, 16, 27, 16]],
                [[2, 14, 27, 14]],
            ],
            dtype=np.int32,
        )

    original_line = cv2.line

    def capture_line(img, pt1, pt2, color, thickness=1):
        drawn.append((pt1, pt2, thickness))
        return original_line(img, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    out, diagnostics = hough_gap_fill_fwhm_sampled(
        mask,
        hough_threshold=1,
        min_line_length=1,
        max_line_gap=20,
        max_thickness=10,
        width_samples=5,
        max_width_search=10,
        return_diagnostics=True,
    )

    assert len(drawn) == 1
    assert drawn[0][2] == 5
    assert diagnostics["raw_line_count"] == 3
    assert diagnostics["cluster_count"] == 1
    assert diagnostics["clusters"][0]["member_count"] == 3
    assert diagnostics["clusters"][0]["sampled_widths"]
    assert out[15, 15] == 255


def test_contour_width_draws_one_centerline_for_duplicate_hough_lines(monkeypatch):
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[13:18, 2:10] = 255
    mask[13:18, 20:28] = 255
    drawn = []

    def fake_lines(*args, **kwargs):
        return np.array(
            [
                [[2, 15, 27, 15]],
                [[2, 16, 27, 16]],
                [[2, 14, 27, 14]],
            ],
            dtype=np.int32,
        )

    original_line = cv2.line

    def capture_line(img, pt1, pt2, color, thickness=1):
        drawn.append((pt1, pt2, thickness))
        return original_line(img, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    out, diagnostics = hough_gap_fill_contour_width(
        mask,
        hough_threshold=1,
        min_line_length=1,
        max_line_gap=20,
        max_thickness=10,
        return_diagnostics=True,
    )

    assert len(drawn) == 1
    assert drawn[0][2] > 1
    assert diagnostics["raw_line_count"] == 3
    assert diagnostics["cluster_count"] == 1
    assert diagnostics["clusters"][0]["member_count"] == 3
    assert diagnostics["clusters"][0]["contour_thickness"] > 1
    assert out[15, 15] == 255


def test_postprocess_segmentation_can_return_diagnostics(monkeypatch):
    pred = np.zeros((40, 40), dtype=np.float32)
    pred[18:23, 5:15] = 1
    pred[18:23, 25:35] = 1

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]], [[5, 21, 34, 21]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    out, diagnostics = postprocess_segmentation(
        pred,
        gap_fill_mode="fwhm_sampled",
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=3,
        min_component_size=20,
        max_thickness=10,
        return_diagnostics=True,
    )

    assert out.shape == pred.shape
    assert diagnostics["mode"] == "fwhm_sampled"
    assert diagnostics["raw_line_count"] == 2
    assert diagnostics["cluster_count"] == 1
    assert diagnostics["clusters"][0]["final_thickness"] > 1


def test_invalid_gap_fill_mode_raises_value_error():
    with pytest.raises(ValueError, match="one_pixel"):
        postprocess_segmentation(np.zeros((10, 10), dtype=np.uint8), gap_fill_mode="wide")

import numpy as np
import cv2

from satellite_trail_segmentation.postprocess.hough import hough_transform


def test_hough_transform_no_lines_returns_binary_image():
    pred = np.zeros((8, 8), dtype=np.float32)
    out = hough_transform(pred, hough_threshold=10, min_length=5, max_gap=1, binary_threshold=0.5)
    np.testing.assert_array_equal(out, np.zeros((8, 8), dtype=np.uint8))


def test_hough_transform_draws_detected_line(monkeypatch):
    pred = np.zeros((8, 8), dtype=np.float32)
    pred[3, 1:7] = 1.0

    drawn = {}

    def fake_lines(*args, **kwargs):
        return np.array([[[1, 3, 6, 3]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(img, pt1, pt2, color, thickness=1):
        drawn["pt1"] = pt1
        drawn["pt2"] = pt2
        drawn["color"] = color
        drawn["thickness"] = thickness
        return original_line(img, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    out = hough_transform(pred, hough_threshold=1, min_length=1, max_gap=1, binary_threshold=0.5)
    assert out.dtype == np.uint8
    assert out[3].sum() > 0
    assert drawn["pt1"] == (1, 3)
    assert drawn["pt2"] == (6, 3)
    assert drawn["color"] == 255

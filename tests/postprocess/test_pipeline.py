import cv2
import numpy as np
import pytest

from satellite_trail_segmentation.postprocess.gap_fill import sampled_width_for_line
from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation


def gapped_stripe_mask():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[18:23, 5:15] = 255
    mask[18:23, 25:35] = 255
    return mask


@pytest.mark.parametrize(
    ("line_mode", "width_mode"),
    [
        ("asta", "none"),
        ("asta", "contour_width"),
        ("asta", "median_sampled_width"),
        ("centerline", "contour_width"),
        ("centerline", "median_sampled_width"),
    ],
)
def test_public_modes_do_not_mutate_input(monkeypatch, line_mode, width_mode):
    mask = gapped_stripe_mask()
    original = mask.copy()

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]], [[5, 21, 34, 21]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    postprocess_segmentation(
        mask,
        line_mode=line_mode,
        width_mode=width_mode,
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        width_samples=5,
        max_width_search=10,
    )

    np.testing.assert_array_equal(mask, original)


def test_sampled_width_recovers_known_synthetic_stripe_width():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[17:24, 5:45] = 255

    width = sampled_width_for_line(mask, np.array([5, 20, 44, 20]), num_samples=7, max_width_search=12)

    assert width == 7


def test_sampled_gap_fills_only_synthetic_gap(monkeypatch):
    mask = gapped_stripe_mask()
    drawn = []

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(image, pt1, pt2, color, thickness=1):
        drawn.append((pt1, pt2, thickness))
        return original_line(image, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    result = postprocess_segmentation(
        mask,
        line_mode="asta",
        width_mode="median_sampled_width",
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        width_samples=9,
        max_width_search=10,
    )

    assert drawn[0][2] == 1
    assert drawn[-1][0][0] > 5
    assert drawn[-1][1][0] < 34
    assert drawn[-1][2] == 5
    assert result[20, 20] == 255
    assert result[20, 40] == 0


def test_sampled_gap_respects_min_fill_gap(monkeypatch):
    mask = gapped_stripe_mask()
    drawn = []

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(image, pt1, pt2, color, thickness=1):
        drawn.append((pt1, pt2, thickness))
        return original_line(image, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    result = postprocess_segmentation(
        mask,
        line_mode="asta",
        width_mode="median_sampled_width",
        min_line_length=1,
        max_line_gap=20,
        min_fill_gap=11,
        morph_kernel_size=1,
        min_component_size=1,
        width_samples=9,
        max_width_search=10,
    )

    assert len(drawn) == 1
    assert result[20, 20] == 255
    assert result[18, 20] == 0


def test_centerline_sampled_width_draws_one_line_per_cluster(monkeypatch):
    mask = gapped_stripe_mask()
    drawn = []

    def fake_lines(*args, **kwargs):
        return np.array(
            [
                [[5, 20, 34, 20]],
                [[5, 21, 34, 21]],
                [[5, 19, 34, 19]],
            ],
            dtype=np.int32,
        )

    original_line = cv2.line

    def capture_line(image, pt1, pt2, color, thickness=1):
        drawn.append((pt1, pt2, thickness))
        return original_line(image, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    result = postprocess_segmentation(
        mask,
        line_mode="centerline",
        width_mode="median_sampled_width",
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        width_samples=5,
        max_width_search=10,
    )

    assert len(drawn) == 1
    assert drawn[0][2] == 5
    assert result[20, 20] == 255


def test_invalid_options_raise_value_error():
    with pytest.raises(ValueError, match="line_mode"):
        postprocess_segmentation(np.zeros((10, 10), dtype=np.uint8), line_mode="wide")
    with pytest.raises(ValueError, match="width_mode"):
        postprocess_segmentation(np.zeros((10, 10), dtype=np.uint8), width_mode="wide")
    with pytest.raises(ValueError, match="none"):
        postprocess_segmentation(np.zeros((10, 10), dtype=np.uint8), line_mode="centerline", width_mode="none")

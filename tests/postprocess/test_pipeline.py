import cv2
import numpy as np
import pytest

from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation


def gapped_stripe_mask():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[18:23, 5:15] = 255
    mask[18:23, 25:35] = 255
    return mask


@pytest.mark.parametrize("mode", ["asta", "contour_gap", "sampled_centerline", "sampled_gap"])
def test_public_modes_do_not_mutate_input(monkeypatch, mode):
    mask = gapped_stripe_mask()
    original = mask.copy()

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]], [[5, 21, 34, 21]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    postprocess_segmentation(
        mask,
        gap_fill_mode=mode,
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        width_samples=5,
        max_width_search=10,
    )

    np.testing.assert_array_equal(mask, original)


def test_sampled_centerline_draws_one_median_thickness_line_per_cluster(monkeypatch):
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

    result, diagnostics = postprocess_segmentation(
        mask,
        gap_fill_mode="sampled_centerline",
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        width_samples=5,
        max_width_search=10,
        return_diagnostics=True,
    )

    assert len(drawn) == 1
    assert drawn[0][2] == 5
    assert diagnostics["cluster_count"] == 1
    assert diagnostics["final_thicknesses"] == [5]
    assert result[20, 20] == 255


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

    result, diagnostics = postprocess_segmentation(
        mask,
        gap_fill_mode="sampled_gap",
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        width_samples=9,
        max_width_search=10,
        return_diagnostics=True,
    )

    assert len(drawn) == 1
    assert drawn[0][0][0] > 5
    assert drawn[0][1][0] < 34
    assert drawn[0][2] == 5
    assert result[20, 20] == 255
    assert result[20, 40] == 0
    assert diagnostics["gap_segments"]


def test_contour_gap_uses_contour_width_for_gap_only_fill(monkeypatch):
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

    result, diagnostics = postprocess_segmentation(
        mask,
        gap_fill_mode="contour_gap",
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        max_thickness=10,
        return_diagnostics=True,
    )

    assert drawn[0][2] == 1
    assert drawn[-1][2] > 1
    assert drawn[-1][0][0] > 5
    assert drawn[-1][1][0] < 34
    assert result[20, 20] == 255
    assert diagnostics["contour_records"]
    assert diagnostics["gap_segments"]


def test_invalid_gap_fill_mode_raises_value_error():
    with pytest.raises(ValueError, match="asta.*contour_gap.*sampled_centerline.*sampled_gap"):
        postprocess_segmentation(np.zeros((10, 10), dtype=np.uint8), gap_fill_mode="wide")


def test_return_diagnostics_includes_line_cluster_width_and_gap_records(monkeypatch):
    mask = gapped_stripe_mask()

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]], [[5, 21, 34, 21]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    result, diagnostics = postprocess_segmentation(
        mask,
        gap_fill_mode="sampled_gap",
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        width_samples=5,
        max_width_search=10,
        return_diagnostics=True,
    )

    assert result.shape == mask.shape
    assert diagnostics["mode"] == "sampled_gap"
    assert diagnostics["hough_line_count"] == 2
    assert diagnostics["cluster_count"] == 1
    assert diagnostics["representative_centerlines"]
    assert diagnostics["sampled_widths"][0]
    assert diagnostics["final_thicknesses"] == [5]
    assert diagnostics["gap_segments"]

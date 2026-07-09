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
    "mode",
    [
        "asta_only",
        "asta_gap_fill",
    ],
)
def test_public_modes_do_not_mutate_input(monkeypatch, mode):
    mask = gapped_stripe_mask()
    original = mask.copy()

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]], [[5, 21, 34, 21]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    postprocess_segmentation(
        mask,
        mode=mode,
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        contour_filter=False,
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
        return np.array([[[5, 20, 14, 20]], [[25, 20, 34, 20]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(image, pt1, pt2, color, thickness=1):
        drawn.append((pt1, pt2, thickness))
        return original_line(image, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    result = postprocess_segmentation(
        mask,
        mode="asta_gap_fill",
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        contour_filter=False,
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
        return np.array([[[5, 20, 14, 20]], [[25, 20, 34, 20]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(image, pt1, pt2, color, thickness=1):
        drawn.append((pt1, pt2, thickness))
        return original_line(image, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    result = postprocess_segmentation(
        mask,
        mode="asta_gap_fill",
        min_line_length=1,
        max_line_gap=20,
        min_fill_gap=11,
        morph_kernel_size=1,
        min_component_size=1,
        contour_filter=False,
        width_samples=9,
        max_width_search=10,
    )

    assert len(drawn) == 2
    assert result[20, 20] == 0
    assert result[18, 20] == 0


def test_asta_gap_fill_uses_second_cluster_pass(monkeypatch):
    mask = gapped_stripe_mask()
    mask[:, 35:] = 0
    mask[18:23, 35:45] = 255
    drawn = []

    def fake_lines(*args, **kwargs):
        return np.array(
            [
                [[5, 20, 14, 20]],
                [[35, 20, 44, 20]],
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
        mode="asta_gap_fill",
        min_line_length=1,
        max_line_gap=60,
        line_cluster_max_along_gap=10,
        max_extension_ratio=5,
        max_fill_gap=30,
        morph_kernel_size=1,
        min_component_size=1,
        contour_filter=False,
        width_samples=5,
        max_width_search=10,
    )

    assert len(drawn) == 3
    assert result[20, 20] == 255


def test_asta_gap_fill_splits_distant_collinear_fragments(monkeypatch):
    mask = np.zeros((40, 130), dtype=np.uint8)
    mask[18:23, 5:35] = 255
    mask[18:23, 90:120] = 255
    drawn = []

    def fake_lines(*args, **kwargs):
        return np.array(
            [
                [[5, 20, 34, 20]],
                [[90, 20, 119, 20]],
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
        mode="asta_gap_fill",
        min_line_length=1,
        max_line_gap=100,
        line_cluster_max_along_gap=25,
        morph_kernel_size=1,
        min_component_size=1,
        contour_filter=False,
        width_samples=5,
        max_width_search=10,
    )

    assert len(drawn) == 2
    assert result[20, 20] == 255
    assert result[20, 104] == 255
    assert result[20, 60] == 0


def test_invalid_options_raise_value_error():
    with pytest.raises(ValueError, match="mode"):
        postprocess_segmentation(np.zeros((10, 10), dtype=np.uint8), mode="wide")


def test_default_return_is_mask_only(monkeypatch):
    mask = gapped_stripe_mask()

    def fake_lines(*args, **kwargs):
        return None

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    result = postprocess_segmentation(mask, morph_kernel_size=1, min_component_size=1, contour_filter=False)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == mask.shape
    assert sorted(np.unique(result).tolist()) == [0, 255]


def test_default_pipeline_uses_paper_style_filtering(monkeypatch):
    mask = np.zeros((140, 160), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    mask[40:110, 40:120] = 255

    def fake_lines(*args, **kwargs):
        return None

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    result = postprocess_segmentation(mask, morph_kernel_size=1)

    assert result[10, 10] == 0
    assert result[70, 70] == 255


def test_contour_details_return_tuple_from_final_mask(monkeypatch):
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[18:23, 5:35] = 255
    mask[2:4, 2:4] = 255

    def fake_lines(*args, **kwargs):
        return None

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    result, details = postprocess_segmentation(
        mask,
        morph_kernel_size=1,
        min_component_size=10,
        contour_filter=False,
        contour_details=True,
        contour_min_area=10,
    )

    assert result.dtype == np.uint8
    assert result.shape == mask.shape
    assert sorted(np.unique(result).tolist()) == [0, 255]
    assert details["contour_count"] == 1
    assert details["contours"][0]["bbox_x"] == 5

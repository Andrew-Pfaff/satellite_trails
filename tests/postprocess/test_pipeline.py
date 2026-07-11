import cv2
import inspect
import numpy as np

from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation


def test_pipeline_defaults_match_asta_implementation():
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(postprocess_segmentation).parameters.items()
    }

    assert defaults["foreground_value"] == 255
    assert defaults["hough_threshold"] == 50
    assert defaults["min_line_length"] == 100
    assert defaults["max_line_gap"] == 250
    assert defaults["morph_kernel_size"] == 3
    assert defaults["min_component_size"] == 500
    assert defaults["contour_filter"] is True
    assert defaults["contour_area_threshold"] == 3000
    assert defaults["contour_hough_threshold"] == 50
    assert defaults["contour_min_line_length"] == 100
    assert defaults["contour_max_line_gap"] == 10
    assert defaults["contour_dbscan_eps"] == 5
    assert defaults["contour_dbscan_min_samples"] == 1
    assert defaults["contour_max_orientation_clusters"] == 5


def gapped_stripe_mask():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[18:23, 5:15] = 255
    mask[18:23, 25:35] = 255
    return mask


def test_pipeline_does_not_mutate_input(monkeypatch):
    mask = gapped_stripe_mask()
    original = mask.copy()

    def fake_lines(*args, **kwargs):
        return np.array([[[5, 20, 34, 20]], [[5, 21, 34, 21]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)

    postprocess_segmentation(
        mask,
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        contour_filter=False,
    )

    np.testing.assert_array_equal(mask, original)


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

import cv2
import numpy as np
import pytest
import torch

from satellite_trail_segmentation.postprocess.hough import (
    cluster_hough_lines,
    line_records,
    representative_centerline,
    standardize_binary_mask,
    to_numpy_2d,
)


def test_to_numpy_2d_handles_numpy_arrays_without_mutation():
    prediction = np.array([[[0.1, 0.9], [0.2, 0.8]]], dtype=np.float32)
    original = prediction.copy()

    result = to_numpy_2d(prediction)

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(prediction, original)


def test_to_numpy_2d_handles_torch_tensors():
    prediction = torch.tensor([[[0.1, 0.9], [0.2, 0.8]]])

    result = to_numpy_2d(prediction)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32))


def test_to_numpy_2d_rejects_ambiguous_shapes():
    with pytest.raises(ValueError, match="2D"):
        to_numpy_2d(np.zeros((2, 3, 4), dtype=np.float32))


def test_standardize_binary_mask_preserves_shape_dtype_and_foreground_value():
    mask = np.array([[0, 1, 7], [0, 0, 3]], dtype=np.int16)

    result = standardize_binary_mask(mask, foreground_value=127)

    assert result.shape == mask.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.array([[0, 127, 127], [0, 0, 127]], dtype=np.uint8))


def test_asta_draws_every_hough_line_with_thickness_one(monkeypatch):
    from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation

    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[15, 2:10] = 255
    drawn = []

    def fake_lines(*args, **kwargs):
        return np.array([[[2, 15, 20, 15]], [[22, 15, 35, 15]]], dtype=np.int32)

    original_line = cv2.line

    def capture_line(image, pt1, pt2, color, thickness=1):
        drawn.append((pt1, pt2, thickness))
        return original_line(image, pt1, pt2, color, thickness=thickness)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_lines)
    monkeypatch.setattr(cv2, "line", capture_line)

    result = postprocess_segmentation(
        mask,
        line_mode="asta",
        width_mode="none",
        min_line_length=1,
        max_line_gap=20,
        morph_kernel_size=1,
        min_component_size=1,
        contour_filter=False,
    )

    assert [line[2] for line in drawn] == [1, 1]
    assert result[15, 20] == 255
    assert result[15, 35] == 255


def test_duplicate_parallel_hough_detections_cluster_into_one_trail():
    lines = np.array(
        [
            [[2, 15, 37, 15]],
            [[2, 16, 37, 16]],
            [[2, 14, 37, 14]],
            [[2, 5, 37, 5]],
        ],
        dtype=np.int32,
    )

    records = line_records(lines)
    clusters = cluster_hough_lines(records, angle_degrees=3, distance=3)

    assert sorted(len(cluster) for cluster in clusters) == [1, 3]


def test_distant_collinear_hough_detections_split_by_along_gap():
    lines = np.array(
        [
            [[5, 15, 35, 15]],
            [[90, 15, 120, 15]],
        ],
        dtype=np.int32,
    )

    records = line_records(lines)
    clusters = cluster_hough_lines(records, angle_degrees=3, distance=3, max_along_gap=25)

    assert sorted(len(cluster) for cluster in clusters) == [1, 1]


def test_nearby_collinear_hough_detections_cluster_by_along_gap():
    lines = np.array(
        [
            [[5, 15, 35, 15]],
            [[50, 15, 80, 15]],
        ],
        dtype=np.int32,
    )

    records = line_records(lines)
    clusters = cluster_hough_lines(records, angle_degrees=3, distance=3, max_along_gap=25)

    assert [len(cluster) for cluster in clusters] == [2]


def test_representative_centerline_is_centered_between_cluster_members():
    lines = np.array([[[5, 10, 35, 10]], [[5, 14, 35, 14]]], dtype=np.int32)

    records = line_records(lines)
    cluster = cluster_hough_lines(records, angle_degrees=3, distance=5)[0]
    centerline = representative_centerline(cluster, image_shape=(30, 40))

    assert centerline.tolist() == [5, 12, 35, 12]

import cv2
import numpy as np
import pytest

from satellite_trail_segmentation.postprocess.contour import (
    clamp_thickness,
    contour_thickness_records,
    estimate_contour_thickness,
    find_contours,
    thickness_for_line,
)


def rotated_box_contour(center=(50, 50), size=(40, 6), angle=0):
    box = cv2.boxPoints((center, size, angle))
    return np.round(box).astype(np.int32).reshape(-1, 1, 2)


def test_find_contours_detects_external_components():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:10, 5:15] = 255
    mask[20:25, 20:25] = 255

    contours = find_contours(mask)

    assert len(contours) == 2


@pytest.mark.parametrize("angle", [0, 45, 90])
def test_estimate_contour_thickness_uses_short_rotated_box_side(angle):
    contour = rotated_box_contour(size=(40, 6), angle=angle)

    thickness = estimate_contour_thickness(contour, min_thickness=1, max_thickness=10)

    assert 5 <= thickness <= 7


def test_estimate_contour_thickness_clamps_extreme_width():
    contour = rotated_box_contour(size=(40, 20), angle=0)

    thickness = estimate_contour_thickness(contour, min_thickness=1, max_thickness=5)

    assert thickness == 5


def test_estimate_contour_thickness_can_skip_upper_clamp():
    contour = rotated_box_contour(size=(40, 20), angle=0)

    thickness = estimate_contour_thickness(contour, min_thickness=1, max_thickness=None)

    assert thickness == 20


def test_estimate_contour_thickness_handles_empty_contour():
    thickness = estimate_contour_thickness(np.empty((0, 1, 2), dtype=np.int32), fallback_thickness=2)

    assert thickness == 2


def test_clamp_thickness_validates_bounds():
    with pytest.raises(ValueError):
        clamp_thickness(1, min_thickness=0)

    with pytest.raises(ValueError):
        clamp_thickness(1, min_thickness=3, max_thickness=2)


def test_thickness_for_line_uses_nearby_contour_record():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[18:23, 5:20] = 255
    records = contour_thickness_records(mask, min_thickness=1, max_thickness=10)

    thickness = thickness_for_line(
        np.array([5, 20, 19, 20], dtype=np.int32),
        records,
        fallback_thickness=1,
        max_contour_distance=5,
    )

    assert thickness > 1


def test_thickness_for_line_falls_back_when_no_contour_is_near():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:6, 2:12] = 255
    records = contour_thickness_records(mask, min_thickness=1, max_thickness=10)

    thickness = thickness_for_line(
        np.array([25, 30, 35, 30], dtype=np.int32),
        records,
        fallback_thickness=1,
        max_contour_distance=5,
    )

    assert thickness == 1

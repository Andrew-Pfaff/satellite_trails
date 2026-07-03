import cv2
import numpy as np
import pytest

from satellite_trail_segmentation.postprocess.contour import (
    clamp_thickness,
    extract_satellite_trail_data,
    find_contours,
    thickness_for_line,
)
from satellite_trail_segmentation.postprocess.contour import (
    contour_thickness_records,
    estimate_contour_thickness,
)


def test_find_contours_detects_external_components():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:10, 5:20] = 255
    mask[25:32, 22:35] = 255

    contours = find_contours(mask)

    assert len(contours) == 2


def test_contour_thickness_helpers_use_nearby_width():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[18:24, 5:35] = 255
    records = contour_thickness_records(mask, min_thickness=1, max_thickness=10)

    thickness = thickness_for_line(np.array([5, 20, 34, 20]), records, max_contour_distance=5)

    assert thickness > 1


def test_estimate_contour_thickness_clamps_and_validates():
    contour = cv2.boxPoints(((20, 20), (30, 8), 20)).round().astype(np.int32).reshape(-1, 1, 2)

    assert estimate_contour_thickness(contour, min_thickness=1, max_thickness=6) == 6
    assert clamp_thickness(4.7, min_thickness=1, max_thickness=None) == 5

    with pytest.raises(ValueError):
        clamp_thickness(1, min_thickness=0)
    with pytest.raises(ValueError):
        clamp_thickness(1, min_thickness=5, max_thickness=4)


def test_extract_satellite_trail_data_returns_expected_geometry_fields():
    mask = np.zeros((60, 90), dtype=np.uint8)
    mask[28:33, 10:70] = 255

    records = extract_satellite_trail_data(mask, min_area=10)

    assert len(records) == 1
    record = records[0]
    assert record["trail_id"] == 1
    assert record["area_px"] == 300
    assert record["centroid_x"] == pytest.approx(39.5)
    assert record["centroid_y"] == pytest.approx(30.0)
    assert record["bbox_x"] == 10
    assert record["bbox_y"] == 28
    assert record["bbox_width"] == 60
    assert record["bbox_height"] == 5
    assert record["length_px"] == pytest.approx(59, abs=2)
    assert record["width_px"] == pytest.approx(5, abs=1)
    assert record["angle_degrees"] == pytest.approx(0, abs=2)
    assert record["endpoint_x1"] < record["endpoint_x2"]


def test_extract_satellite_trail_data_includes_optional_image_intensity_fields():
    mask = np.zeros((50, 70), dtype=np.uint8)
    mask[20:25, 10:55] = 255
    image = np.ones_like(mask, dtype=np.float32) * 2
    image[mask > 0] = 10

    records = extract_satellite_trail_data(mask, image=image, min_area=10, dilation_size=7)

    record = records[0]
    assert record["trail_pixel_count"] == 225
    assert record["trail_flux_sum"] == pytest.approx(2250)
    assert record["trail_intensity_mean"] == pytest.approx(10)
    assert record["trail_intensity_median"] == pytest.approx(10)
    assert record["local_background_mean"] == pytest.approx(2)
    assert record["local_background_std"] == pytest.approx(0)
    assert record["snr_proxy"] > 0

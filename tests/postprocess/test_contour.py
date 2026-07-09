import numpy as np
import pytest

from satellite_trail_segmentation.postprocess.contour import (
    extract_contour_details,
    contour_filtering,
)


def test_extract_contour_details_returns_expected_geometry_fields():
    mask = np.zeros((60, 90), dtype=np.uint8)
    mask[28:33, 10:70] = 255

    details = extract_contour_details(mask, min_area=10)

    assert details["contour_count"] == 1
    record = details["contours"][0]
    assert record["trail_id"] == 1
    assert record["area_px"] == pytest.approx(236)
    assert record["centroid_x"] == pytest.approx(39.5)
    assert record["centroid_y"] == pytest.approx(30)
    assert record["bbox_x"] == 10
    assert record["bbox_y"] == 28
    assert record["bbox_width"] == 60
    assert record["bbox_height"] == 5
    assert record["length_px"] == pytest.approx(59)
    assert record["width_px"] == pytest.approx(4)
    assert record["angle_degrees"] == pytest.approx(0, abs=1e-6)
    assert record["endpoint_x1"] != record["endpoint_x2"]


def test_extract_contour_details_filters_small_contours():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:7, 5:7] = 255
    mask[10:16, 10:20] = 255

    details = extract_contour_details(mask, min_area=10)

    assert details["contour_count"] == 1


def test_extract_contour_details_handles_empty_mask():
    details = extract_contour_details(np.zeros((20, 20), dtype=np.uint8), min_area=10)

    assert details == {"contours": [], "contour_count": 0}


def test_contour_filter_removes_small_contours():
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    mask[30:80, 30:95] = 255

    result = contour_filtering(mask, area_threshold=1000)

    assert result[7, 7] == 0
    assert result[50, 50] == 255

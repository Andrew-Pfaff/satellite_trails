import numpy as np

from satellite_trail_segmentation.postprocess.contour import contour_width_for_line, contour_widths


def test_contour_widths_measure_external_components():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[18:24, 5:35] = 255

    records = contour_widths(mask)

    assert len(records) == 1
    assert records[0]["width"] > 1


def test_contour_width_for_line_uses_nearest_contour():
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[10:14, 5:45] = 255
    mask[40:52, 5:45] = 255
    records = contour_widths(mask)

    width = contour_width_for_line(np.array([5, 46, 44, 46]), records, max_distance=10)

    assert width >= 10


def test_contour_width_for_line_uses_fallback_when_no_contour_is_close():
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[10:14, 5:45] = 255
    records = contour_widths(mask)

    width = contour_width_for_line(np.array([5, 50, 44, 50]), records, fallback_width=3, max_distance=5)

    assert width == 3

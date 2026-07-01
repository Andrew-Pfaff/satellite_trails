import cv2
import numpy as np

from satellite_trail_segmentation.postprocess.contour import (
    contour_thickness_records,
    thickness_for_line,
)
from satellite_trail_segmentation.postprocess.hough import (
    morphological_close,
    remove_small_components,
    standardize_binary_mask,
    to_numpy_2d,
)


def hough_gap_fill_contour_width(
    mask,
    hough_threshold=50,
    min_line_length=100,
    max_line_gap=250,
    min_thickness=1,
    max_thickness=5,
    fallback_thickness=1,
    max_contour_distance=20,
):
    """
    Fills Hough line gaps using thickness estimated from nearby contours.

    Contour widths are estimated from the pre-Hough binary mask. Each detected
    Hough line is matched to the nearest contour and drawn with that contour's
    estimated thickness. Lines without a nearby contour use ``fallback_thickness``.
    """

    binary = standardize_binary_mask(mask)
    output = binary.copy()
    records = contour_thickness_records(
        binary,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        fallback_thickness=fallback_thickness,
    )

    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            thickness = thickness_for_line(
                line[0],
                records,
                fallback_thickness=fallback_thickness,
                max_contour_distance=max_contour_distance,
            )
            cv2.line(output, (x1, y1), (x2, y2), 255, thickness=thickness)

    return standardize_binary_mask(output)


def postprocess_segmentation_contour_width(
    prediction,
    foreground_value=255,
    hough_threshold=50,
    min_line_length=100,
    max_line_gap=250,
    morph_kernel_size=3,
    min_component_size=500,
    min_thickness=1,
    max_thickness=5,
    fallback_thickness=1,
    max_contour_distance=20,
):
    """
    Runs experimental contour-width-aware segmentation postprocessing.

    The pipeline is: binary mask standardization, adaptive-thickness Hough gap
    filling, morphological closing, and small-component cleanup.
    """

    arr = to_numpy_2d(prediction)
    mask = standardize_binary_mask(arr, foreground_value=foreground_value)
    mask = hough_gap_fill_contour_width(
        mask,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        fallback_thickness=fallback_thickness,
        max_contour_distance=max_contour_distance,
    )
    mask = morphological_close(mask, kernel_size=morph_kernel_size)
    mask = remove_small_components(mask, min_size=min_component_size)

    if foreground_value != 255:
        mask = ((mask > 0).astype(np.uint8) * foreground_value).astype(np.uint8)

    return mask

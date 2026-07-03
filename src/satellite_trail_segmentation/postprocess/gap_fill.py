import numpy as np

from satellite_trail_segmentation.postprocess.postprocess_utils import standardize_binary_mask
from satellite_trail_segmentation.postprocess.contour import contour_width_for_line
from satellite_trail_segmentation.postprocess.hough import draw_centerlines


def sampled_width_for_line(mask, line, num_samples=9, max_width_search=25, fallback_width=1):
    """
    Estimates line width from perpendicular binary-mask samples.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask.
        line (array-like): Line coordinates as (x1, y1, x2, y2).
        num_samples (int): Number of samples along the line. Defaults to 9.
        max_width_search (int): Perpendicular search radius in pixels. Defaults to 25.
        fallback_width (float): Width used when no samples are valid. Defaults to 1.

    Returns:
        width (float): Median sampled width.
    """

    # Skeleton implementation: actual perpendicular sampling will be filled in later.
    del mask, line, num_samples, max_width_search
    return float(fallback_width)


def widths_for_centerlines(
    mask, centerlines, width_mode, contour_records=None, width_samples=9,
    max_width_search=25, fallback_width=1,
):
    """
    Chooses drawing widths for representative centerlines.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask.
        centerlines (list): Representative centerline arrays.
        width_mode (str): One of "contour_width" or "median_sampled_width".
        contour_records (list): Optional contour width records.
        width_samples (int): Number of samples for sampled-width mode. Defaults to 9.
        max_width_search (int): Perpendicular search radius for sampled-width mode. Defaults to 25.
        fallback_width (float): Width used when estimation fails. Defaults to 1.

    Returns:
        widths (list): Width values, one per centerline.
    """

    if width_mode == "contour_width":
        return [
            contour_width_for_line(line, contour_records or [], fallback_width=fallback_width)
            for line in centerlines
        ]
    if width_mode == "median_sampled_width":
        return [
            sampled_width_for_line(
                mask, line, num_samples=width_samples, max_width_search=max_width_search,
                fallback_width=fallback_width,
            )
            for line in centerlines
        ]

    raise ValueError("width_mode must be 'contour_width' or 'median_sampled_width'")


def fill_gaps_along_lines(
    mask, centerlines, width_mode, contour_records=None, width_samples=9,
    max_width_search=25, fallback_width=1, foreground_value=255,
):
    """
    Draws gap-fill segments along centerlines using the requested width mode.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask.
        centerlines (list): Candidate centerline arrays.
        width_mode (str): One of "contour_width" or "median_sampled_width".
        contour_records (list): Optional contour width records.
        width_samples (int): Number of samples for sampled-width mode. Defaults to 9.
        max_width_search (int): Perpendicular search radius for sampled-width mode. Defaults to 25.
        fallback_width (float): Width used when estimation fails. Defaults to 1.
        foreground_value (int): Foreground value to draw. Defaults to 255.

    Returns:
        output (np.ndarray): uint8 mask with gap fills drawn.
    """

    # Skeleton implementation: draws full centerlines for now. We will replace this
    # with true gap-only segment detection before using it for final experiments.
    output = standardize_binary_mask(mask, foreground_value=foreground_value)
    widths = widths_for_centerlines(
        output, centerlines, width_mode, contour_records=contour_records,
        width_samples=width_samples, max_width_search=max_width_search,
        fallback_width=fallback_width,
    )
    return draw_centerlines(output, centerlines, widths, foreground_value=foreground_value)

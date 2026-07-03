from satellite_trail_segmentation.postprocess.postprocess_utils import (
    morphological_close,
    remove_small_components,
    standardize_binary_mask,
)
from satellite_trail_segmentation.postprocess.contour import contour_widths, extract_contour_details
from satellite_trail_segmentation.postprocess.gap_fill import fill_gaps_along_lines, widths_for_centerlines
from satellite_trail_segmentation.postprocess.hough import cluster_hough_lines, detect_hough_lines, draw_centerlines, draw_hough_lines, line_records, representative_centerline


LINE_MODES = ("asta", "centerline")
WIDTH_MODES = ("none", "contour_width", "median_sampled_width")


def postprocess_segmentation(
    binary_mask_prediction,
    *,
    line_mode="asta",
    width_mode="none",
    foreground_value=255,
    hough_threshold=50,
    min_line_length=100,
    max_line_gap=250,
    line_cluster_angle_degrees=3,
    line_cluster_distance=8,
    width_samples=9,
    max_width_search=25,
    max_contour_distance=20,
    min_fill_gap=10,
    fallback_width=1,
    morph_kernel_size=3,
    min_component_size=100,
    contour_details=False,
    contour_min_area=10,
):
    """
    Postprocesses a binary segmentation mask using a selected Hough drawing strategy.

    Args:
        binary_mask_prediction (np.ndarray or torch.Tensor): Binary-like prediction mask.
        line_mode (str): Hough drawing mode. One of "asta" or "centerline". Defaults to "asta".
        width_mode (str): Width mode. One of "none", "contour_width", or "median_sampled_width". Defaults to "none".
        foreground_value (int): Foreground value for returned mask. Defaults to 255.
        hough_threshold (int): Minimum Hough accumulator threshold. Defaults to 50.
        min_line_length (int): Minimum accepted Hough line length in pixels. Defaults to 100.
        max_line_gap (int): Maximum connected line gap in pixels. Defaults to 250.
        line_cluster_angle_degrees (float): Maximum cluster orientation difference. Defaults to 3.
        line_cluster_distance (float): Maximum cluster perpendicular distance in pixels. Defaults to 8.
        width_samples (int): Number of width samples for median sampled width. Defaults to 9.
        max_width_search (int): Perpendicular search radius for median sampled width. Defaults to 25.
        max_contour_distance (float): Maximum line-to-contour distance for contour width. Defaults to 20.
        min_fill_gap (int): Minimum gap length to fill for ASTA width modes. Defaults to 10.
        fallback_width (float): Width used when estimation fails. Defaults to 1.
        morph_kernel_size (int): Morphological closing kernel size. Defaults to 3.
        min_component_size (int): Minimum connected component size to keep. Defaults to 100.
        contour_details (bool): Whether to return final-mask contour details. Defaults to False.
        contour_min_area (float): Minimum contour area included in contour details. Defaults to 10.

    Returns:
        output (np.ndarray or tuple): uint8 postprocessed binary mask, or ``(mask, contour_details)`` when contour_details is True.
    """

    if line_mode not in LINE_MODES:
        raise ValueError(f"line_mode must be one of {LINE_MODES}, got {line_mode!r}")
    if width_mode not in WIDTH_MODES:
        raise ValueError(f"width_mode must be one of {WIDTH_MODES}, got {width_mode!r}")
    if line_mode == "centerline" and width_mode == "none":
        raise ValueError("width_mode='none' is only valid with line_mode='asta'")

    mask = standardize_binary_mask(binary_mask_prediction, foreground_value=foreground_value)
    lines = detect_hough_lines(
        mask,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )

    if line_mode == "asta":
        output = draw_hough_lines(mask, lines, thickness=1, foreground_value=foreground_value)
        if width_mode != "none":
            records = line_records(lines)
            clusters = cluster_hough_lines(
                records,
                angle_degrees=line_cluster_angle_degrees,
                distance=line_cluster_distance,
            )
            centerlines = [representative_centerline(cluster, mask.shape) for cluster in clusters]
            contour_records = contour_widths(mask) if width_mode == "contour_width" else None
            output = fill_gaps_along_lines(
                output,
                centerlines,
                width_mode,
                contour_records=contour_records,
                width_samples=width_samples,
                max_width_search=max_width_search,
                max_contour_distance=max_contour_distance,
                fallback_width=fallback_width,
                support_mask=mask,
                min_gap=min_fill_gap,
                max_gap=max_line_gap,
                foreground_value=foreground_value,
            )
    else:
        records = line_records(lines)
        clusters = cluster_hough_lines(
            records,
            angle_degrees=line_cluster_angle_degrees,
            distance=line_cluster_distance,
        )
        centerlines = [representative_centerline(cluster, mask.shape) for cluster in clusters]
        contour_records = contour_widths(mask) if width_mode == "contour_width" else None
        widths = widths_for_centerlines(
            mask,
            centerlines,
            width_mode,
            contour_records=contour_records,
            width_samples=width_samples,
            max_width_search=max_width_search,
            max_contour_distance=max_contour_distance,
            fallback_width=fallback_width,
        )
        output = draw_centerlines(mask, centerlines, widths, foreground_value=foreground_value)

    output = morphological_close(output, kernel_size=morph_kernel_size, foreground_value=foreground_value)
    output = remove_small_components(
        output,
        min_component_size=min_component_size,
        foreground_value=foreground_value,
    )
    output = standardize_binary_mask(output, foreground_value=foreground_value)
    if contour_details:
        return output, extract_contour_details(output, min_area=contour_min_area)
    return output

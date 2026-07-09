from satellite_trail_segmentation.postprocess.postprocess_utils import (
    morphological_close,
    remove_small_components,
    standardize_binary_mask,
)
from satellite_trail_segmentation.postprocess.contour import contour_filtering, extract_contour_details
from satellite_trail_segmentation.postprocess.gap_fill import fill_gaps_along_lines
from satellite_trail_segmentation.postprocess.hough import cluster_hough_lines, detect_hough_lines, draw_hough_lines, line_records, representative_centerline


POSTPROCESS_MODES = ("asta_only", "asta_gap_fill")


def postprocess_segmentation(
    binary_mask_prediction,
    *,
    mode="asta_only",
    foreground_value=255,
    hough_threshold=50,
    min_line_length=100,
    max_line_gap=250,
    line_cluster_angle_degrees=3,
    line_cluster_distance=8,
    line_cluster_max_along_gap=250,
    max_extension_ratio=1.5,
    width_samples=9,
    max_width_search=25,
    min_fill_gap=10,
    max_fill_gap=250,
    fallback_width=1,
    morph_kernel_size=3,
    min_component_size=500,
    contour_filter=True,
    contour_area_threshold=3000,
    contour_details=False,
    contour_min_area=10,
):
    """
    Postprocesses a binary segmentation mask using one of two fixed ASTA configurations.

    Args:
        binary_mask_prediction (np.ndarray or torch.Tensor): Binary-like prediction mask.
        mode (str): Postprocessing mode. One of "asta_only" or "asta_gap_fill". Defaults to "asta_only".
        foreground_value (int): Foreground value for returned mask. Defaults to 255.
        hough_threshold (int): Minimum Hough accumulator threshold. Defaults to 50.
        min_line_length (int): Minimum accepted Hough line length in pixels. Defaults to 100.
        max_line_gap (int): Maximum connected line gap in pixels. Defaults to 250.
        line_cluster_angle_degrees (float): Maximum cluster orientation difference for gap-fill mode. Defaults to 3.
        line_cluster_distance (float): Maximum cluster perpendicular distance in pixels for gap-fill mode. Defaults to 8.
        line_cluster_max_along_gap (float or None): Maximum gap between Hough segments along the fitted trail direction before they are split into separate clusters in gap-fill mode. Defaults to 250.
        max_extension_ratio (float): Maximum representative-centerline span relative to observed cluster endpoint span in gap-fill mode. Defaults to 1.5.
        width_samples (int): Number of width samples for gap-fill mode. Defaults to 9.
        max_width_search (int): Perpendicular search radius for gap-fill mode. Defaults to 25.
        min_fill_gap (int): Minimum gap length to fill in gap-fill mode. Defaults to 10.
        max_fill_gap (int): Maximum gap length to fill in gap-fill mode. Defaults to 250.
        fallback_width (float): Width used when estimation fails. Defaults to 1.
        morph_kernel_size (int): Morphological closing kernel size. Defaults to 3.
        min_component_size (int): Minimum connected component size to keep. Defaults to 500.
        contour_filter (bool): Whether to apply ASTA-style contour filtering. Defaults to True.
        contour_area_threshold (float): Minimum contour area kept by contour filtering. Defaults to 3000.
        contour_details (bool): Whether to return final-mask contour details. Defaults to False.
        contour_min_area (float): Minimum contour area included in contour details. Defaults to 10.

    Returns:
        output (np.ndarray or tuple): uint8 postprocessed binary mask, or ``(mask, contour_details)`` when contour_details is True.
    """

    if mode not in POSTPROCESS_MODES:
        raise ValueError(f"mode must be one of {POSTPROCESS_MODES}, got {mode!r}")

    mask = standardize_binary_mask(binary_mask_prediction, foreground_value=foreground_value)
    lines = detect_hough_lines(
        mask,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )
    output = draw_hough_lines(mask, lines, thickness=1, foreground_value=foreground_value)

    if mode == "asta_gap_fill":
        records = line_records(lines)
        clusters = cluster_hough_lines(
            records,
            angle_degrees=line_cluster_angle_degrees,
            distance=line_cluster_distance,
            max_along_gap=line_cluster_max_along_gap,
        )
        first_pass_centerlines = [representative_centerline(cluster, mask.shape, max_extension_ratio=max_extension_ratio) for cluster in clusters]
        centerline_records = line_records(first_pass_centerlines)
        merged_clusters = cluster_hough_lines(
            centerline_records,
            angle_degrees=line_cluster_angle_degrees,
            distance=line_cluster_distance,
            max_along_gap=line_cluster_max_along_gap,
        )
        centerlines = [representative_centerline(cluster, mask.shape, max_extension_ratio=max_extension_ratio) for cluster in merged_clusters]
        output = fill_gaps_along_lines(
            output,
            centerlines,
            width_samples=width_samples,
            max_width_search=max_width_search,
            fallback_width=fallback_width,
            support_mask=output,
            min_gap=min_fill_gap,
            max_gap=max_fill_gap,
            foreground_value=foreground_value,
        )

    output = morphological_close(output, kernel_size=morph_kernel_size, foreground_value=foreground_value)
    output = remove_small_components(
        output,
        min_component_size=min_component_size,
        foreground_value=foreground_value,
    )
    if contour_filter:
        output = contour_filtering(output, area_threshold=contour_area_threshold, foreground_value=foreground_value)
    output = standardize_binary_mask(output, foreground_value=foreground_value)
    if contour_details:
        return output, extract_contour_details(output, min_area=contour_min_area)
    return output

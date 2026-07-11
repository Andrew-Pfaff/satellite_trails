from satellite_trail_segmentation.postprocess.contour import contour_filtering, extract_contour_details
from satellite_trail_segmentation.postprocess.hough import detect_hough_lines, draw_hough_lines
from satellite_trail_segmentation.postprocess.postprocess_utils import (
    morphological_close,
    remove_small_components,
    standardize_binary_mask,
)


def postprocess_segmentation(
    binary_mask_prediction,
    *,
    foreground_value=255,
    hough_threshold=50,
    min_line_length=100,
    max_line_gap=250,
    morph_kernel_size=3,
    min_component_size=500,
    contour_filter=True,
    contour_area_threshold=3000,
    contour_hough_threshold=50,
    contour_min_line_length=100,
    contour_max_line_gap=10,
    contour_dbscan_eps=5,
    contour_dbscan_min_samples=1,
    contour_max_orientation_clusters=5,
    contour_details=False,
    contour_min_area=10,
):
    """
    Postprocesses a binary segmentation mask with the ASTA-style Hough pipeline.

    Args:
        binary_mask_prediction (np.ndarray or torch.Tensor): Binary-like prediction mask.
        foreground_value (int): Foreground value for returned mask. Defaults to 255.
        hough_threshold (int): Minimum Hough accumulator threshold. Defaults to 50.
        min_line_length (int): Minimum accepted Hough line length in pixels. Defaults to 100.
        max_line_gap (int): Maximum connected line gap in pixels. Defaults to 250.
        morph_kernel_size (int): Morphological closing kernel size. Defaults to 3.
        min_component_size (int): Minimum connected component size to keep. Defaults to 500.
        contour_filter (bool): Whether to apply ASTA-style contour filtering. Defaults to True.
        contour_area_threshold (float): Minimum contour area kept by contour filtering. Defaults to 3000.
        contour_hough_threshold (int): Accumulator threshold for ASTA's contour-local Hough pass. Defaults to 50.
        contour_min_line_length (int): Minimum contour-local Hough line length. Defaults to 100.
        contour_max_line_gap (int): Maximum contour-local Hough line gap. Defaults to 10.
        contour_dbscan_eps (float): DBSCAN angle-neighbourhood radius in degrees. Defaults to 5.
        contour_dbscan_min_samples (int): DBSCAN minimum samples per angle cluster. Defaults to 1.
        contour_max_orientation_clusters (int): Angle-cluster count at which ASTA rejects a contour. Defaults to 5.
        contour_details (bool): Whether to return final-mask contour details. Defaults to False.
        contour_min_area (float): Minimum contour area included in contour details. Defaults to 10.

    Returns:
        output (np.ndarray or tuple): uint8 postprocessed binary mask, or ``(mask, contour_details)`` when contour_details is True.
    """

    mask = standardize_binary_mask(binary_mask_prediction, foreground_value=foreground_value)
    lines = detect_hough_lines(
        mask,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )
    output = draw_hough_lines(mask, lines, thickness=1, foreground_value=foreground_value)
    output = morphological_close(output, kernel_size=morph_kernel_size, foreground_value=foreground_value)
    output = remove_small_components(
        output,
        min_component_size=min_component_size,
        foreground_value=foreground_value,
    )
    if contour_filter:
        output = contour_filtering(
            output,
            area_threshold=contour_area_threshold,
            foreground_value=foreground_value,
            hough_threshold=contour_hough_threshold,
            min_line_length=contour_min_line_length,
            max_line_gap=contour_max_line_gap,
            dbscan_eps=contour_dbscan_eps,
            dbscan_min_samples=contour_dbscan_min_samples,
            max_orientation_clusters=contour_max_orientation_clusters,
        )
    output = standardize_binary_mask(output, foreground_value=foreground_value)

    if contour_details:
        return output, extract_contour_details(output, min_area=contour_min_area)
    return output

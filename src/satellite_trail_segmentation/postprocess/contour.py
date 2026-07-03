import cv2
import numpy as np

from satellite_trail_segmentation.postprocess.postprocess_utils import standardize_binary_mask


def contour_widths(mask):
    """
    Measures contour widths using min-area rectangles.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask.

    Returns:
        records (list): Contour width records.
    """

    binary = standardize_binary_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    records = []

    for index, contour in enumerate(contours):
        if len(contour) < 2:
            continue
        rect = cv2.minAreaRect(contour)
        width = float(min(rect[1])) if rect[1][0] > 0 and rect[1][1] > 0 else 1.0
        records.append(
            {
                "index": index,
                "contour": contour,
                "center": rect[0],
                "width": width,
                "area": float(cv2.contourArea(contour)),
            }
        )

    return records


def contour_width_for_line(line, contour_records, fallback_width=1):
    """
    Selects a contour-derived width for one line.

    Args:
        line (array-like): Line coordinates as (x1, y1, x2, y2).
        contour_records (list): Records returned by contour_widths.
        fallback_width (float): Width used when no contour is available. Defaults to 1.

    Returns:
        width (float): Selected contour width.
    """

    # Skeleton implementation: nearest-contour matching will be filled in later.
    del line
    if not contour_records:
        return float(fallback_width)
    return float(np.median([record["width"] for record in contour_records]))

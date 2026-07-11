import cv2
import numpy as np

from satellite_trail_segmentation.postprocess.postprocess_utils import standardize_binary_mask


def detect_hough_lines(mask, hough_threshold=50, min_line_length=100, max_line_gap=250):
    """
    Detects probabilistic Hough line segments in a binary mask.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask.
        hough_threshold (int): Minimum Hough accumulator threshold. Defaults to 50.
        min_line_length (int): Minimum accepted Hough line length in pixels. Defaults to 100.
        max_line_gap (int): Maximum connected line gap in pixels. Defaults to 250.

    Returns:
        lines (np.ndarray or None): OpenCV HoughLinesP output.
    """

    binary = standardize_binary_mask(mask)
    return cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )


def draw_hough_lines(mask, lines, thickness=1, foreground_value=255):
    """
    Draws every Hough line segment onto a copy of a mask.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask to copy before drawing.
        lines (np.ndarray or None): OpenCV HoughLinesP output.
        thickness (int): OpenCV line thickness. Defaults to 1.
        foreground_value (int): Foreground value to draw. Defaults to 255.

    Returns:
        output (np.ndarray): uint8 binary mask with lines drawn.
    """

    output = standardize_binary_mask(mask, foreground_value=foreground_value)
    if lines is None:
        return output

    thickness = max(1, int(round(float(thickness))))
    height, width = output.shape
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        x1 = int(np.clip(round(x1), 0, width - 1))
        x2 = int(np.clip(round(x2), 0, width - 1))
        y1 = int(np.clip(round(y1), 0, height - 1))
        y2 = int(np.clip(round(y2), 0, height - 1))
        cv2.line(output, (x1, y1), (x2, y2), int(foreground_value), thickness=thickness)

    return standardize_binary_mask(output, foreground_value=foreground_value)

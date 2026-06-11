import cv2
import numpy as np


def to_numpy_2d(prediction):
    """
    Converts a segmentation prediction to a 2D NumPy array.

    Accepts NumPy arrays and torch tensors. Singleton dimensions are removed
    only when the result is unambiguous, such as (1, H, W) or (H, W, 1).
    """

    if hasattr(prediction, "detach") and hasattr(prediction, "cpu"):
        prediction = prediction.detach().cpu().numpy()
    else:
        prediction = np.asarray(prediction)

    prediction = np.array(prediction, copy=True)
    if prediction.ndim == 2:
        return prediction

    squeezed = np.squeeze(prediction)
    if squeezed.ndim != 2:
        raise ValueError(f"prediction must resolve to a 2D array, got shape {prediction.shape}")

    return squeezed


def binarize_prediction(prediction, threshold=0.58, foreground_value=255):
    """
    Converts a probability map or binary-like array into a uint8 mask.

    Returns an array with values 0 and ``foreground_value``.
    """

    prediction = to_numpy_2d(prediction)
    if foreground_value <= 0 or foreground_value > 255:
        raise ValueError("foreground_value must be in the range 1..255")

    unique_values = np.unique(prediction)
    has_zero = unique_values.size > 0 and np.isclose(unique_values[0], 0)
    max_value = unique_values[-1] if unique_values.size > 0 else 0
    binary_like = (
        unique_values.size <= 2
        and has_zero
        and (
            np.issubdtype(prediction.dtype, np.integer)
            or np.isclose(max_value, 1)
            or np.isclose(max_value, foreground_value)
            or np.isclose(max_value, 255)
        )
    )
    if binary_like:
        binary = prediction > 0
    else:
        binary = prediction > threshold

    return (binary.astype(np.uint8) * foreground_value).astype(np.uint8)


def standardize_binary_mask(mask, foreground_value=255):
    """
    Converts an already-binary mask into a uint8 mask for OpenCV.

    Any nonzero value is treated as foreground. This does not threshold
    probability maps; thresholding should happen before postprocessing.
    """

    mask = to_numpy_2d(mask)
    if foreground_value <= 0 or foreground_value > 255:
        raise ValueError("foreground_value must be in the range 1..255")

    return ((mask > 0).astype(np.uint8) * foreground_value).astype(np.uint8)


def hough_gap_fill(mask, thickness, hough_threshold=50, min_line_length=100, max_line_gap=250):
    """
    Fills gaps in a binary segmentation mask using probabilistic Hough lines.

    Detected line segments are drawn onto a copy of ``mask`` with the provided
    thickness.
    """

    mask = standardize_binary_mask(mask)
    output = mask.copy()
    thickness = max(1, int(round(thickness)))

    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), 255, thickness=thickness)

    return output


def morphological_close(mask, kernel_size=3):
    """
    Applies morphological closing to fill small holes and local gaps.
    """

    mask = standardize_binary_mask(mask)
    if kernel_size <= 1:
        return mask.copy()

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return standardize_binary_mask(closed)


def remove_small_components(mask, min_size=500):
    """
    Removes connected foreground components smaller than ``min_size`` pixels.
    """

    mask = standardize_binary_mask(mask)
    if min_size <= 1:
        return mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            output[labels == label] = 255

    return output


def postprocess_segmentation(prediction, foreground_value=255, hough_threshold=50, min_line_length=100, max_line_gap=250, morph_kernel_size=3, min_component_size=500):
    """
    Runs the full segmentation postprocessing pipeline.

    The steps are: binary mask standardization, one-pixel Hough gap filling,
    morphological closing, and small-component cleanup.
    """

    arr = to_numpy_2d(prediction)
    mask = standardize_binary_mask(arr, foreground_value=foreground_value)
    mask = hough_gap_fill(
        mask,
        thickness=1,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )
    mask = morphological_close(mask, kernel_size=morph_kernel_size)
    mask = remove_small_components(mask, min_size=min_component_size)

    if foreground_value != 255:
        mask = ((mask > 0).astype(np.uint8) * foreground_value).astype(np.uint8)

    return mask

import cv2
import numpy as np


def to_numpy_2d(mask):
    """
    Converts a mask object to a copied 2D NumPy array.

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask input.

    Returns:
        array (np.ndarray): Copied 2D NumPy array.
    """

    if hasattr(mask, "detach") and hasattr(mask, "cpu"):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)

    array = np.array(array, copy=True)
    if array.ndim == 2:
        return array

    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise ValueError(f"mask must resolve to a 2D array, got shape {array.shape}")

    return squeezed


def standardize_binary_mask(mask, foreground_value=255):
    """
    Converts a binary-like mask to uint8 background/foreground values.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask. Any nonzero value is foreground.
        foreground_value (int): Foreground value for the returned mask. Defaults to 255.

    Returns:
        binary (np.ndarray): uint8 mask containing only 0 and foreground_value.
    """

    if foreground_value <= 0 or foreground_value > 255:
        raise ValueError("foreground_value must be in the range 1..255")

    array = to_numpy_2d(mask)
    return ((array > 0).astype(np.uint8) * int(foreground_value)).astype(np.uint8)


def morphological_close(mask, kernel_size=3, foreground_value=255):
    """
    Applies morphological closing to a binary mask.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask.
        kernel_size (int): Square kernel size. Values <= 1 return a copy. Defaults to 3.
        foreground_value (int): Foreground value for the returned mask. Defaults to 255.

    Returns:
        closed (np.ndarray): uint8 closed binary mask.
    """

    binary = standardize_binary_mask(mask, foreground_value=foreground_value)
    if kernel_size <= 1:
        return binary.copy()

    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return standardize_binary_mask(closed, foreground_value=foreground_value)


def remove_small_components(mask, min_component_size=100, foreground_value=255):
    """
    Removes connected foreground components smaller than min_component_size.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask.
        min_component_size (int): Minimum connected component area to keep. Defaults to 100.
        foreground_value (int): Foreground value for the returned mask. Defaults to 255.

    Returns:
        cleaned (np.ndarray): uint8 binary mask with small components removed.
    """

    binary = standardize_binary_mask(mask, foreground_value=foreground_value)
    if min_component_size <= 1:
        return binary.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_size:
            cleaned[labels == label] = int(foreground_value)

    return cleaned

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

    binary = standardize_binary_mask(mask)
    x1, y1, x2, y2 = np.asarray(line).reshape(4).astype(float)
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0:
        return float(fallback_width)

    perpendicular = np.array([-dy / length, dx / length], dtype=float)
    offsets = np.arange(-int(max_width_search), int(max_width_search) + 1)
    center_index = int(max_width_search)
    widths = []

    for x, y in zip(np.linspace(x1, x2, max(2, int(num_samples))), np.linspace(y1, y2, max(2, int(num_samples)))):
        xs = np.rint(x + offsets * perpendicular[0]).astype(int)
        ys = np.rint(y + offsets * perpendicular[1]).astype(int)
        in_bounds = (xs >= 0) & (xs < binary.shape[1]) & (ys >= 0) & (ys < binary.shape[0])
        profile = np.zeros(offsets.shape, dtype=bool)
        profile[in_bounds] = binary[ys[in_bounds], xs[in_bounds]] > 0

        if not np.any(profile):
            continue

        padded = np.concatenate(([False], profile, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        runs = [(int(start), int(end - 1)) for start, end in zip(changes[0::2], changes[1::2])]
        start, end = min(
            runs,
            key=lambda run: 0 if run[0] <= center_index <= run[1] else min(abs(center_index - run[0]), abs(center_index - run[1])),
        )
        widths.append(end - start + 1)

    if not widths:
        return float(fallback_width)
    return float(np.median(widths))


def widths_for_centerlines(
    mask, centerlines, width_mode, contour_records=None, width_samples=9,
    max_width_search=25, fallback_width=1, max_contour_distance=20,
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
        max_contour_distance (float): Maximum line-to-contour distance. Defaults to 20.

    Returns:
        widths (list): Width values, one per centerline.
    """

    if width_mode == "contour_width":
        return [
            contour_width_for_line(
                line, contour_records or [], fallback_width=fallback_width,
                max_distance=max_contour_distance, num_samples=width_samples,
            )
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
    max_width_search=25, fallback_width=1, foreground_value=255, support_mask=None,
    min_gap=10, max_gap=250, max_contour_distance=20,
):
    """
    Draws gap-fill segments along centerlines using the requested width mode.

    The ``mask`` argument is the drawing target: gap segments are added to a
    copy of this mask and returned. The optional ``support_mask`` argument is
    the mask used to decide which parts of each centerline are already covered
    by the original prediction. When ``support_mask`` is None, ``mask`` is used
    for both drawing and support detection.

    For ASTA-plus-gap filling, call this with ``mask`` set to the ASTA output
    and ``support_mask`` set to the original binary prediction. This keeps the
    one-pixel Hough result while detecting gaps from the pre-Hough prediction.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask to draw gap segments onto.
        centerlines (list): Candidate centerline arrays.
        width_mode (str): One of "contour_width" or "median_sampled_width".
        contour_records (list): Optional contour width records.
        width_samples (int): Number of samples for sampled-width mode. Defaults to 9.
        max_width_search (int): Perpendicular search radius for sampled-width mode. Defaults to 25.
        fallback_width (float): Width used when estimation fails. Defaults to 1.
        foreground_value (int): Foreground value to draw. Defaults to 255.
        support_mask (np.ndarray or torch.Tensor): Optional mask used to identify existing foreground support and gaps.
        min_gap (int): Minimum gap length to fill. Defaults to 10.
        max_gap (int): Maximum gap length to fill. Defaults to 250.
        max_contour_distance (float): Maximum line-to-contour distance. Defaults to 20.

    Returns:
        output (np.ndarray): uint8 mask with gap fills drawn.
    """

    output = standardize_binary_mask(mask, foreground_value=foreground_value)
    support = standardize_binary_mask(mask if support_mask is None else support_mask)
    widths = widths_for_centerlines(
        support, centerlines, width_mode, contour_records=contour_records,
        width_samples=width_samples, max_width_search=max_width_search,
        fallback_width=fallback_width, max_contour_distance=max_contour_distance,
    )

    gap_lines = []
    gap_widths = []

    for line, width in zip(centerlines, widths):
        x1, y1, x2, y2 = np.asarray(line).reshape(4).astype(float)
        length = int(np.ceil(np.hypot(x2 - x1, y2 - y1)))
        if length < 2:
            continue

        xs = np.rint(np.linspace(x1, x2, length + 1)).astype(int)
        ys = np.rint(np.linspace(y1, y2, length + 1)).astype(int)
        samples = np.column_stack([xs, ys])
        keep = np.ones(len(samples), dtype=bool)
        keep[1:] = np.any(samples[1:] != samples[:-1], axis=1)
        samples = samples[keep]

        dx = x2 - x1
        dy = y2 - y1
        line_length = float(np.hypot(dx, dy))
        if line_length == 0:
            continue

        perpendicular = np.array([-dy / line_length, dx / line_length], dtype=float)
        radius = max(1, int(np.ceil(float(width) / 2)))
        offsets = np.arange(-radius, radius + 1)
        supported = []

        for x, y in samples:
            check_xs = np.rint(float(x) + offsets * perpendicular[0]).astype(int)
            check_ys = np.rint(float(y) + offsets * perpendicular[1]).astype(int)
            in_bounds = (
                (check_xs >= 0) & (check_xs < support.shape[1])
                & (check_ys >= 0) & (check_ys < support.shape[0])
            )
            supported.append(bool(np.any(support[check_ys[in_bounds], check_xs[in_bounds]] > 0)))

        supported = np.asarray(supported, dtype=bool)
        missing = ~supported
        padded = np.concatenate(([False], missing, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])

        for start, end_exclusive in zip(changes[0::2], changes[1::2]):
            end = int(end_exclusive - 1)
            start = int(start)
            gap_length = end - start + 1
            if start == 0 or end == len(samples) - 1:
                continue
            if gap_length < min_gap:
                continue
            if gap_length > max_gap:
                continue
            if not supported[start - 1] or not supported[end + 1]:
                continue

            x_start, y_start = samples[start]
            x_end, y_end = samples[end]
            gap_lines.append(np.array([x_start, y_start, x_end, y_end], dtype=np.int32))
            gap_widths.append(width)

    if not gap_lines:
        return output
    return draw_centerlines(output, gap_lines, gap_widths, foreground_value=foreground_value)

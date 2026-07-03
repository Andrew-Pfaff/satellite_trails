import cv2
import numpy as np

from satellite_trail_segmentation.postprocess.postprocess_utils import standardize_binary_mask, to_numpy_2d


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
        binary, rho=1, theta=np.pi / 180, threshold=hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
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


def line_records(lines):
    """
    Converts Hough lines into simple geometry records for trail grouping.

    Args:
        lines (np.ndarray or None): OpenCV HoughLinesP output.

    Returns:
        records (list): List of line records.
    """

    if lines is None:
        return []

    records = []
    for index, line in enumerate(np.asarray(lines).reshape(-1, 4)):
        x1, y1, x2, y2 = line.astype(float)
        dx = x2 - x1
        dy = y2 - y1
        angle = float(np.arctan2(dy, dx) % np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        normal = np.array([-direction[1], direction[0]], dtype=float)
        midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
        records.append(
            {
                "index": index,
                "line": np.array([x1, y1, x2, y2], dtype=float),
                "angle": angle,
                "direction": direction,
                "normal": normal,
                "midpoint": midpoint,
                "offset": float(np.dot(midpoint, normal)),
                "length": float(np.hypot(dx, dy)),
            }
        )

    return records


def cluster_hough_lines(records, angle_degrees=3, distance=8):
    """
    Groups Hough line records that likely describe the same physical trail.

    Args:
        records (list): Line records from line_records.
        angle_degrees (float): Maximum orientation difference in degrees. Defaults to 3.
        distance (float): Maximum perpendicular offset difference in pixels. Defaults to 8.

    Returns:
        clusters (list): List of record lists.
    """

    if not records:
        return []

    angle_threshold = np.deg2rad(float(angle_degrees))
    visited = set()
    clusters = []

    for start_index in range(len(records)):
        if start_index in visited:
            continue

        queue = [start_index]
        visited.add(start_index)
        cluster = []

        while queue:
            current_index = queue.pop(0)
            current = records[current_index]
            cluster.append(current)

            for candidate_index, candidate in enumerate(records):
                if candidate_index in visited:
                    continue

                angle_diff = abs(current["angle"] - candidate["angle"]) % np.pi
                angle_diff = min(angle_diff, np.pi - angle_diff)
                midpoint_delta = candidate["midpoint"] - current["midpoint"]
                distance_a = abs(float(np.dot(midpoint_delta, current["normal"])))
                distance_b = abs(float(np.dot(-midpoint_delta, candidate["normal"])))

                if angle_diff <= angle_threshold and min(distance_a, distance_b) <= distance:
                    visited.add(candidate_index)
                    queue.append(candidate_index)

        clusters.append(cluster)

    return clusters


def representative_centerline(cluster, image_shape):
    """
    Builds one representative centerline for a Hough cluster.

    Args:
        cluster (list): One cluster of line records.
        image_shape (tuple): Output image shape as (height, width).

    Returns:
        line (np.ndarray): Integer centerline coordinates as (x1, y1, x2, y2).
    """

    if not cluster:
        raise ValueError("cluster must contain at least one line record")

    angles = np.asarray([record["angle"] for record in cluster], dtype=float) % np.pi
    reference = angles[0]
    shifted_angles = ((angles - reference + np.pi / 2) % np.pi) - np.pi / 2
    angle = float((reference + np.median(shifted_angles)) % np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)

    endpoints = []
    offsets = []
    for record in cluster:
        x1, y1, x2, y2 = record["line"]
        endpoints.append(np.array([x1, y1], dtype=float))
        endpoints.append(np.array([x2, y2], dtype=float))
        offsets.append(float(np.dot(record["midpoint"], normal)))

    offset = float(np.median(offsets))
    projections = [float(np.dot(point, direction)) for point in endpoints]
    start = offset * normal + min(projections) * direction
    end = offset * normal + max(projections) * direction
    line = np.array([start[0], start[1], end[0], end[1]], dtype=float)

    height, width = image_shape
    x1, y1, x2, y2 = np.rint(line).astype(int)
    return np.array(
        [
            int(np.clip(x1, 0, width - 1)),
            int(np.clip(y1, 0, height - 1)),
            int(np.clip(x2, 0, width - 1)),
            int(np.clip(y2, 0, height - 1)),
        ],
        dtype=np.int32,
    )


def draw_centerlines(mask, centerlines, thicknesses, foreground_value=255):
    """
    Draws representative centerlines with matching thickness values.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask to copy before drawing.
        centerlines (list): List of integer centerline arrays.
        thicknesses (list): List of line thickness values.
        foreground_value (int): Foreground value to draw. Defaults to 255.

    Returns:
        output (np.ndarray): uint8 binary mask with centerlines drawn.
    """

    output = standardize_binary_mask(mask, foreground_value=foreground_value)
    for line, thickness in zip(centerlines, thicknesses):
        x1, y1, x2, y2 = np.asarray(line).reshape(4).astype(int)
        thickness = max(1, int(round(float(thickness))))
        cv2.line(output, (x1, y1), (x2, y2), int(foreground_value), thickness=thickness)

    return standardize_binary_mask(output, foreground_value=foreground_value)

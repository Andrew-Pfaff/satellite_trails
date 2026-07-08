import cv2
import numpy as np
from sklearn.cluster import DBSCAN

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


def contour_width_for_line(line, contour_records, fallback_width=1, max_distance=20, num_samples=9):
    """
    Selects a contour-derived width for one line.

    Args:
        line (array-like): Line coordinates as (x1, y1, x2, y2).
        contour_records (list): Records returned by contour_widths.
        fallback_width (float): Width used when no contour is available. Defaults to 1.
        max_distance (float): Maximum allowed line-to-contour distance. Defaults to 20.
        num_samples (int): Number of line samples used for matching. Defaults to 9.

    Returns:
        width (float): Selected contour width.
    """

    if not contour_records:
        return float(fallback_width)

    x1, y1, x2, y2 = np.asarray(line).reshape(4).astype(float)
    xs = np.linspace(x1, x2, max(2, int(num_samples)))
    ys = np.linspace(y1, y2, max(2, int(num_samples)))
    best_record = None
    best_distance = float("inf")

    for record in contour_records:
        distances = [
            abs(cv2.pointPolygonTest(record["contour"], (float(x), float(y)), True))
            for x, y in zip(xs, ys)
        ]
        distance = float(np.min(distances))
        if distance < best_distance:
            best_distance = distance
            best_record = record

    if best_record is None or best_distance > max_distance:
        return float(fallback_width)
    return float(best_record["width"])


def contour_filtering(mask, area_threshold=3000, foreground_value=255):
    """
    Applies contour filtering to a binary mask.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask after Hough filling and cleanup.
        area_threshold (float): Minimum contour area to keep. Defaults to 3000.
        foreground_value (int): Foreground value for returned mask. Defaults to 255.

    Returns:
        output (np.ndarray): uint8 binary mask after contour filtering.
    """

    output = standardize_binary_mask(mask, foreground_value=foreground_value)
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < area_threshold:
            cv2.drawContours(output, [contour], -1, 0, thickness=cv2.FILLED)
            continue

        contour_mask = np.zeros_like(output, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
        lines = cv2.HoughLinesP(contour_mask, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

        if lines is None:
            continue

        lines = lines[:, 0, :]
        angles = np.array([_line_angle(line) for line in lines])
        clustering = DBSCAN(eps=5, min_samples=1).fit(angles.reshape(-1, 1))
        unique_labels = set(clustering.labels_)

        if len(unique_labels) >= 5:
            cv2.drawContours(output, [contour], -1, 0, thickness=cv2.FILLED)
            continue

        for label in unique_labels:
            if label == -1:
                continue

            cluster_lines = lines[clustering.labels_ == label]
            cluster_mask = np.zeros_like(contour_mask, dtype=np.uint8)
            for x1, y1, x2, y2 in cluster_lines:
                cv2.line(cluster_mask, (x1, y1), (x2, y2), 1, thickness=1)

            cluster_contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cluster_contour in cluster_contours:
                if cv2.contourArea(cluster_contour) < area_threshold:
                    cv2.drawContours(output, [cluster_contour], -1, 0, thickness=cv2.FILLED)

    return standardize_binary_mask(output, foreground_value=foreground_value)


def _line_angle(line):
    """
    Calculates a Hough line segment angle in degrees.

    Args:
        line (array-like): Line coordinates as (x1, y1, x2, y2).

    Returns:
        float: Angle in the range [0, 360).
    """

    x1, y1, x2, y2 = line
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle if angle >= 0 else angle + 360


def extract_contour_details(mask, min_area=10):
    """
    Extracts image-plane contour details from a binary mask.

    Args:
        mask (np.ndarray or torch.Tensor): Binary-like mask.
        min_area (float): Minimum contour area to include. Defaults to 10.

    Returns:
        details (dict): Dictionary containing contour records and contour count.
    """

    binary = standardize_binary_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    records = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            rect_center = cv2.minAreaRect(contour)[0]
            centroid_x = float(rect_center[0])
            centroid_y = float(rect_center[1])
        else:
            centroid_x = float(moments["m10"] / moments["m00"])
            centroid_y = float(moments["m01"] / moments["m00"])

        bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        side_lengths = [
            float(np.hypot(*(box[(index + 1) % 4] - box[index])))
            for index in range(4)
        ]
        long_index = int(np.argmax(side_lengths))
        short_index = (long_index + 1) % 4
        length = side_lengths[long_index]
        width = side_lengths[short_index]
        point_a = box[long_index]
        point_b = box[(long_index + 1) % 4]
        angle = float(np.degrees(np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])))

        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180

        records.append(
            {
                "trail_id": len(records) + 1,
                "area_px": area,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "bbox_x": int(bbox_x),
                "bbox_y": int(bbox_y),
                "bbox_width": int(bbox_width),
                "bbox_height": int(bbox_height),
                "length_px": float(length),
                "width_px": float(width),
                "angle_degrees": angle,
                "endpoint_x1": float(point_a[0]),
                "endpoint_y1": float(point_a[1]),
                "endpoint_x2": float(point_b[0]),
                "endpoint_y2": float(point_b[1]),
            }
        )

    return {"contours": records, "contour_count": len(records)}

import cv2
import numpy as np

from satellite_trail_segmentation.postprocess.hough import standardize_binary_mask


def find_contours(mask):
    """
    Finds external contours in a binary mask.

    Any nonzero mask value is treated as foreground. The returned contours are
    OpenCV contours suitable for ``cv2.minAreaRect`` and ``cv2.pointPolygonTest``.
    """

    binary = standardize_binary_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def clamp_thickness(thickness, min_thickness=1, max_thickness=5):
    """
    Rounds and clamps a line thickness estimate to a valid OpenCV thickness.
    """

    if min_thickness < 1:
        raise ValueError("min_thickness must be at least 1")
    if max_thickness < min_thickness:
        raise ValueError("max_thickness must be greater than or equal to min_thickness")

    return int(np.clip(round(float(thickness)), min_thickness, max_thickness))


def estimate_contour_thickness(contour, min_thickness=1, max_thickness=5, fallback_thickness=1):
    """
    Estimates trail thickness from the shorter side of a contour's rotated box.

    This is intended as a simple, local width estimate for elongated trail-like
    components. If the contour is empty or degenerate, ``fallback_thickness`` is
    returned.
    """

    if contour is None or len(contour) == 0:
        return clamp_thickness(fallback_thickness, min_thickness, max_thickness)

    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    short_side = min(width, height)
    if not np.isfinite(short_side) or short_side <= 0:
        return clamp_thickness(fallback_thickness, min_thickness, max_thickness)

    return clamp_thickness(short_side, min_thickness, max_thickness)


def contour_thickness_records(mask, min_thickness=1, max_thickness=5, fallback_thickness=1):
    """
    Builds contour/thickness records from a binary mask.
    """

    records = []
    for contour in find_contours(mask):
        records.append(
            {
                "contour": contour,
                "thickness": estimate_contour_thickness(
                    contour,
                    min_thickness=min_thickness,
                    max_thickness=max_thickness,
                    fallback_thickness=fallback_thickness,
                ),
            }
        )

    return records


def line_sample_points(line, num_samples=9):
    """
    Samples evenly spaced ``(x, y)`` points along a Hough line segment.
    """

    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")

    x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4].astype(float)
    xs = np.linspace(x1, x2, num_samples)
    ys = np.linspace(y1, y2, num_samples)
    return list(zip(xs, ys))


def contour_distance_to_line(contour, line, num_samples=9):
    """
    Returns the nearest distance between a contour and sampled line points.

    A distance of zero is returned if any sampled point lies inside or on the
    contour. Otherwise the nearest outside distance is returned.
    """

    distances = []
    for point in line_sample_points(line, num_samples=num_samples):
        signed_distance = cv2.pointPolygonTest(contour, point, measureDist=True)
        if signed_distance >= 0:
            return 0.0
        distances.append(abs(signed_distance))

    if not distances:
        return np.inf

    return float(min(distances))


def thickness_for_line(
    line,
    records,
    fallback_thickness=1,
    max_contour_distance=20,
    num_samples=9,
):
    """
    Chooses an adaptive line thickness from the nearest contour record.

    If no contour is close enough to the line, ``fallback_thickness`` is used.
    """

    if not records:
        return max(1, int(round(fallback_thickness)))

    best_record = None
    best_distance = np.inf
    for record in records:
        distance = contour_distance_to_line(record["contour"], line, num_samples=num_samples)
        if distance < best_distance:
            best_distance = distance
            best_record = record

    if best_record is None or best_distance > max_contour_distance:
        return max(1, int(round(fallback_thickness)))

    return int(best_record["thickness"])

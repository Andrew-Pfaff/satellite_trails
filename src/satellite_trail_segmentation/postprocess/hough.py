import numpy as np
import cv2

from satellite_trail_segmentation.unet_model.evaluate import image_threshold


def hough_transform(full_field_pred, hough_threshold, min_length, max_gap, binary_threshold=0.5):
    """
    Applies a probabilistic Hough line transform to a segmentation prediction.

    Binarizes the prediction, detects line segments via the probabilistic Hough transform, and draws detected lines back onto the mask to fill gaps and connect fragmented trail detections.

    Args:
        full_field_pred (np.ndarray): 2D float prediction array from the segmentation model
        hough_threshold (int): Minimum number of votes required to detect a line
        min_length (int): Minimum line length in pixels; shorter segments are discarded
        max_gap (int): Maximum gap in pixels between segments to be joined into one line
        binary_threshold (float): Threshold for binarizing the prediction. Defaults to 0.5

    Returns:
        full_field_pred (np.ndarray): Binarized mask with detected lines drawn in, same shape as input
    """
    
    full_field_pred = image_threshold(full_field_pred, binary_threshold)
    lines = cv2.HoughLinesP(full_field_pred, rho=1, theta=np.pi/180, threshold=hough_threshold, minLineLength=min_length, maxLineGap=max_gap)

    if lines is not None:    
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw a white line (255) with a specific thickness
            cv2.line(full_field_pred, (x1, y1), (x2, y2), 255, thickness=2)

    return full_field_pred
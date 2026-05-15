import numpy as np
import cv2

from satellite_trail_segmentation.unet_model.evaluate import image_threshold


def hough_tranform(full_field_pred, hough_threshold, min_length, max_gap, binary_threshold=0.5):
    full_field_pred = image_threshold(full_field_pred, binary_threshold)
    lines = cv2.HoughLinesP(full_field_pred, rho=1, theta=np.pi/180, threshold=hough_threshold, minLineLength=min_length, maxLineGap=max_gap)

    if lines is not None:    
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw a white line (255) with a specific thickness
            cv2.line(full_field_pred, (x1, y1), (x2, y2), 255, thickness=2)

    return full_field_pred
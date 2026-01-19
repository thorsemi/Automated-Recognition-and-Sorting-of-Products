# classification.py
import cv2
import numpy as np


def classify_front_back(frame, box, white_ratio_threshold: float = 0.75):
    """
    Classify a leaflet as Front or Back based on how white it is.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (color image).
    box : np.ndarray
        4x2 array with the box corners (as returned by cv2.boxPoints).
    white_ratio_threshold : float
        If the ratio of white pixels is above this, the leaflet is classified as Back.

    Returns
    -------
    label : str
        "Front" or "Back".
    ratio : float
        Ratio of white pixels in [0, 1].
    """
    # Mask for the leaflet
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [box], contourIdx=-1, color=255, thickness=-1)

    # Extract leaflet region
    leaflet = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(leaflet, cv2.COLOR_BGR2GRAY)

    # White pixels = very bright
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixels = cv2.countNonZero(white_mask)
    total_pixels = cv2.countNonZero(mask)

    ratio = white_pixels / float(total_pixels) if total_pixels > 0 else 0.0

    if ratio > white_ratio_threshold:
        return "Back", ratio
    else:
        return "Front", ratio

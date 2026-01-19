# classification.py
import cv2
import numpy as np
from typing import Tuple


def classify_front_back(
    frame: np.ndarray,
    box: np.ndarray,
    white_intensity_threshold: int = 220,
    white_ratio_threshold_back: float = 0.85,
) -> Tuple[str, float]:
    """
    Klassificer leaflet som FORSIDE / BAGSIDE baseret på hvor hvid den er.

    Antagelse:
      - Bagside: næsten helt hvid
      - Forside: mindre hvid (tekst, logo osv.)

    Params:
      white_intensity_threshold : grænse for hvide pixels (0-255)
      white_ratio_threshold_back: hvis andel hvide pixels > dette => 'Back'

    Returnerer:
      label        : 'Front' eller 'Back'
      white_ratio  : andel hvide pixels (0-1) inde i boksen
    """
    # Sørg for gråskala
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Maske for den enkelte leaflet (samme box som du tegner)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [box], contourIdx=-1, color=255, thickness=-1)

    leaflet = cv2.bitwise_and(gray, gray, mask=mask)

    # Tæl hvide pixels
    _, white_mask = cv2.threshold(
        leaflet,
        white_intensity_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    white_pixels = cv2.countNonZero(white_mask)
    total_pixels = cv2.countNonZero(mask)

    white_ratio = white_pixels / float(total_pixels) if total_pixels > 0 else 0.0

    # Simpel regel:
    #   meget hvid  -> Back
    #   ellers      -> Front
    if white_ratio > white_ratio_threshold_back:
        label = "Back"
    else:
        label = "Front"

    return label, white_ratio

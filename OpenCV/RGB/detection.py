# detection.py
import cv2
import numpy as np

# === Parameters for leaflet detection ===
MIN_AREA_SINGLE     = 5000
MAX_AREA_SINGLE     = 30000
MAX_AREA_FRAME      = 200000
ASPECT_MIN_SINGLE   = 2.5     # for brede (tykke) sedler
ASPECT_MAX_SINGLE   = 4.2     # for lange sedler
RECTANGULARITY_MIN  = 0.70
RECTANGULARITY_OVER = 0.85
THRESH_GRAY         = 160
KERNEL_SIZE         = 5

# Fallback area for ONE leaflet if no singles are found
SINGLE_AREA_REF = 35000.0

# Currently not used directly in main, but kept for completeness
DOUBLE_MIN_FACTOR = 1.6
DOUBLE_MAX_FACTOR = 2.6
TRIPLE_MAX_FACTOR = 3.6   # >3.6 = heavy


def detect_leaflets(frame):
    """
    Detect leaflets and potential overlaps using grayscale + geometry.

    Parameters
    ----------
    frame : np.ndarray
        BGR image.

    Returns
    -------
    boxes : list
        List of (box, rect, num_corners) for single leaflets.
    rects : list
        List of cv2.minAreaRect outputs for single leaflets.
    mask : np.ndarray
        Binary mask used for detection.
    valid_contours : list
        Contours considered as single leaflets.
    overlap_contours : list of dict
        Each dict contains:
            - "cnt"          : contour
            - "num_corners"  : corner count (approx poly)
            - "reason"       : why it was classified as overlap
            - "area"         : contour area
            - "area_factor"  : area / single_area_ref
            - "overlap_label": "light", "double", "triple" or "heavy"
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESH_GRAY, 255, cv2.THRESH_BINARY)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    boxes = []
    rects = []
    valid_contours = []

    overlap_candidates = []  # (cnt, num_corners, reason, area)
    single_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_SINGLE or area > MAX_AREA_FRAME:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw == 0 or rh == 0:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if x <= 1 or y <= 1 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1):
            # touching the border → ignore
            continue

        aspect_ratio = max(rw, rh) / max(1e-6, min(rw, rh))
        rect_area = rw * rh
        rectangularity = (area / rect_area) if rect_area > 0 else 0.0

        # --- coarse overlap rules ---
        reason = None
        if area > MAX_AREA_SINGLE:
            reason = f"area>{MAX_AREA_SINGLE}"
        elif aspect_ratio < ASPECT_MIN_SINGLE:
            reason = f"aspect<{ASPECT_MIN_SINGLE}"
        elif aspect_ratio > ASPECT_MAX_SINGLE:
            reason = f"aspect>{ASPECT_MAX_SINGLE}"
        elif rectangularity < RECTANGULARITY_MIN:
            reason = f"rect<{RECTANGULARITY_MIN}"

        if reason is not None:
            overlap_candidates.append((cnt, None, reason, area))
            continue

        # corner check
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_corners = len(approx)

        if num_corners > 6 and rectangularity < RECTANGULARITY_OVER:
            overlap_candidates.append(
                (cnt, num_corners, f"corners>6&rect<{RECTANGULARITY_OVER}", area)
            )
            continue

        # === Valid single leaflet ===
        box = cv2.boxPoints(rect).astype(int)
        boxes.append((box, rect, num_corners))
        rects.append(rect)
        valid_contours.append(cnt)
        single_areas.append(area)

    # --- reference area for overlap analysis ---
    if single_areas:
        single_area_ref = float(np.median(single_areas))
    else:
        single_area_ref = SINGLE_AREA_REF  # fallback

    overlap_contours = []
    for cnt, num_corners, reason, area in overlap_candidates:
        if single_area_ref and single_area_ref > 0:
            factor = area / single_area_ref
        else:
            factor = 0.0

        if factor < DOUBLE_MIN_FACTOR:
            overlap_label = "light"
        elif factor < DOUBLE_MAX_FACTOR:
            overlap_label = "double"
        elif factor < TRIPLE_MAX_FACTOR:
            overlap_label = "triple"
        else:
            overlap_label = "heavy"

        overlap_contours.append(
            {
                "cnt": cnt,
                "num_corners": num_corners,
                "reason": reason,
                "area": area,
                "area_factor": factor,
                "overlap_label": overlap_label,
            }
        )

    return boxes, rects, mask, valid_contours, overlap_contours

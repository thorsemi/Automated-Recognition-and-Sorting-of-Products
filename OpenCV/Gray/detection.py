# detection.py
import cv2
import numpy as np
from typing import Tuple, List, Any

# === Parameters for leaflet detection ===
MIN_AREA_SINGLE     = 5000
MAX_AREA_SINGLE     = 30000
MAX_AREA_FRAME      = 200000
ASPECT_MIN_SINGLE   = 2.5     # for bred (for "tyk")
ASPECT_MAX_SINGLE   = 4.2     # for lang
RECTANGULARITY_MIN  = 0.70
RECTANGULARITY_OVER = 0.85
THRESH_GRAY         = 160
KERNEL_SIZE         = 5


def make_binary_mask(
    frame: np.ndarray,
    thresh: int = THRESH_GRAY
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Laver en binær maske ud fra et BGR- eller gråskala-billede.

    Returnerer:
        gray : gråskala-billede
        mask : binær maske (0 og 255)
    """
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return gray, mask


# Vi beholder den her til evt. debug, men bruger den ikke længere
def is_overlap_via_inner_contours(
    gray: np.ndarray,
    cnt: np.ndarray,
    min_inner_area: float = 2000.0
) -> bool:
    """
    (Ikke i brug som hård regel længere)
    """
    x, y, w, h = cv2.boundingRect(cnt)
    roi = gray[y:y + h, x:x + w]
    _, roi_mask = cv2.threshold(roi, THRESH_GRAY, 255, cv2.THRESH_BINARY)

    inner_contours, _ = cv2.findContours(
        roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    large_inner = [
        c for c in inner_contours if cv2.contourArea(c) > min_inner_area
    ]
    return len(large_inner) >= 2


def detect_leaflets(
    frame: np.ndarray
) -> Tuple[
    List[Tuple[np.ndarray, Any, int]],
    List[Any],
    np.ndarray,
    List[np.ndarray],
    List[Tuple[np.ndarray, int | None, str]]
]:
    """
    Detekter leaflets i et BGR- eller gråskala-frame.

    Returnerer:
        boxes            : liste af (boxPoints, rect, num_corners)
        rects            : liste af rect (minAreaRect)
        mask_det         : binær maske brugt til detektion
        valid_contours   : konturer der blev accepteret som enkelt-leaflets
        overlap_contours : konturer der blev flagget som overlap, med reason
    """
    # Sørg for gråskala
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Threshold + morph for at få en pæn maske til detektion
    _, mask_det = cv2.threshold(gray, THRESH_GRAY, 255, cv2.THRESH_BINARY)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask_det = cv2.morphologyEx(mask_det, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask_det, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape
    boxes: list = []
    rects: list = []
    valid_contours: list = []
    overlap_contours: list = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_SINGLE or area > MAX_AREA_FRAME:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw == 0 or rh == 0:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        # Ignorér ting der rører kanten
        if (
            x <= 1 or y <= 1
            or (x + bw) >= (w - 1)
            or (y + bh) >= (h - 1)
        ):
            continue

        aspect_ratio = max(rw, rh) / min(rw, rh)
        rect_area = rw * rh
        rectangularity = (area / rect_area) if rect_area > 0 else 0.0

        # --- Geometri-baseret overlap ---
        if area > MAX_AREA_SINGLE:
            overlap_contours.append((cnt, None, f"area>{MAX_AREA_SINGLE}"))
            continue
        if aspect_ratio < ASPECT_MIN_SINGLE:
            overlap_contours.append((cnt, None, f"aspect<{ASPECT_MIN_SINGLE}"))
            continue
        if aspect_ratio > ASPECT_MAX_SINGLE:
            overlap_contours.append((cnt, None, f"aspect>{ASPECT_MAX_SINGLE}"))
            continue
        if rectangularity < RECTANGULARITY_MIN:
            overlap_contours.append(
                (cnt, None, f"rect<{RECTANGULARITY_MIN}")
            )
            continue

        # --- Corner-baseret overlap ---
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_corners = len(approx)

        if num_corners > 6 and rectangularity < RECTANGULARITY_OVER:
            overlap_contours.append(
                (cnt, num_corners,
                 f"corners>6&rect<{RECTANGULARITY_OVER}")
            )
            continue

        # --- Gyldig enkelt leaflet ---
        box = cv2.boxPoints(rect).astype(int)
        boxes.append((box, rect, num_corners))
        rects.append(rect)
        valid_contours.append(cnt)

    return boxes, rects, mask_det, valid_contours, overlap_contours

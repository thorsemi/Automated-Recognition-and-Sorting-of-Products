# main_image.py
from pathlib import Path
import cv2
import numpy as np

from detection import detect_leaflets, SINGLE_AREA_REF
from classification import classify_front_back


# =============== LINE DETECTOR (integreret) =============== #

class LineDetector:
    """Handles drawing and simple line hit detection logic."""

    def __init__(self, top_line_y=24, distance_between=779, middle_x=1500, middle_x_2=35):
        self.top_line_y = top_line_y
        self.bottom_line_y = top_line_y + distance_between
        self.middle_x = middle_x
        self.middle_x_2 = middle_x_2

    def draw_lines(self, frame):
        """Draws reference lines on frame and returns annotated copy."""
        frame_display = frame.copy()
        height, width, _ = frame.shape

        # Horizontal lines
        cv2.line(frame_display, (0, self.top_line_y), (width, self.top_line_y), (0, 0, 255), 3)
        cv2.line(frame_display, (0, self.bottom_line_y), (width, self.bottom_line_y), (0, 0, 255), 3)

        # Vertical lines
        cv2.line(frame_display, (self.middle_x, self.top_line_y),
                 (self.middle_x, self.bottom_line_y), (0, 255, 0), 3)
        cv2.line(frame_display, (self.middle_x_2, self.top_line_y),
                 (self.middle_x_2, self.bottom_line_y), (255, 0, 0), 3)

        return frame_display

    def check_collision(self, cx, cy):
        """Checks whether (cx, cy) hits any of the lines and returns an event code."""
        if abs(cy - self.top_line_y) < 10:
            return 3
        elif abs(cy - self.bottom_line_y) < 10:
            return 2
        elif abs(cx - self.middle_x) < 10:
            return 1
        elif abs(cx - self.middle_x_2) < 10:
            return 4
        return None


# ==== PATHS & SETTINGS ====
IMAGE_PATH = r"C:\Users\emilt\Desktop\yolo_dataset\images\train\rrr_000270.png"
OUT_DIR = r"C:\Users\emilt\Desktop\yolo_dataset\images\train"

SHOW_WINDOW = True
AREA_OVERLAP_THRESHOLD = 16000.0


def main() -> None:
    in_path = Path(IMAGE_PATH)
    assert in_path.is_file(), f"Could not find image: {in_path}"

    img = cv2.imread(str(in_path))
    assert img is not None, f"cv2.imread failed on: {in_path}"

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_processed.png"

    boxes, rects, mask, contours, overlap_contours = detect_leaflets(img)
    outlined = img.copy()

    # Line detector setup (samme geometri som i de andre scripts)
    line_detector = LineDetector(
        top_line_y=24,
        distance_between=779,
        middle_x=1500,
        middle_x_2=35,
    )

    # base area from single leaflets
    single_areas = [cv2.contourArea(c) for c in contours]
    if single_areas:
        base_area = float(np.median(single_areas))
    else:
        base_area = SINGLE_AREA_REF

    # normal leaflets (single detections)
    for idx, (box, rect, num_corners) in enumerate(boxes):
        cnt = contours[idx]
        area = cv2.contourArea(cnt)
        cx, cy = rect[0]

        # line events for single leaflets
        event = line_detector.check_collision(cx, cy)
        if event is not None:
            print(f"Line event {event} (x={cx:.0f}, y={cy:.0f})")

        # fallback: large area -> overlap
        if base_area > 0 and area > AREA_OVERLAP_THRESHOLD:
            factor = area / base_area
            cv2.drawContours(outlined, [box], 0, (0, 0, 255), 3)
            x, y, w, h = cv2.boundingRect(cnt)

            line1 = f"OVERLAP x{factor:.1f}"
            if factor > 2.3:
                line2 = "HEAVY (3+ overlap)"
            else:
                line2 = "2_overlap"

            cv2.putText(
                outlined,
                line1,
                (x, max(15, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                outlined,
                line2,
                (x, max(30, y + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                outlined,
                f"A={area:.0f} [area>{int(AREA_OVERLAP_THRESHOLD)}]",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            continue

        # otherwise: treat as single leaflet and classify front/back
        cv2.drawContours(outlined, [box], 0, (0, 255, 0), 2)
        label, ratio = classify_front_back(img, box)

        cv2.putText(
            outlined,
            f"{label} ({ratio:.2f})",
            (int(cx), int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),  # blue
            2,
        )
        cv2.putText(
            outlined,
            f"A={area:.0f}",
            (int(cx), int(cy) + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # overlap from detection (geometric)
    for ov in overlap_contours:
        cnt = ov["cnt"]
        area = ov["area"]
        rect = cv2.minAreaRect(cnt)
        box2 = cv2.boxPoints(rect).astype(int)
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(outlined, [box2], 0, (0, 0, 255), 3)

        # line events for overlap blobs (center of rotated rect)
        (cx, cy), (rw, rh), angle = rect
        event = line_detector.check_collision(cx, cy)
        if event is not None:
            print(f"Line event {event} (overlap) (x={cx:.0f}, y={cy:.0f})")

        if base_area > 0:
            factor = area / base_area
            line1 = f"OVERLAP x{factor:.1f}"
            if factor > 2.3:
                line2 = "HEAVY (3+ overlap)"
            else:
                line2 = "2_overlap"
        else:
            factor = 0.0
            line1 = "OVERLAP"
            line2 = ""

        cv2.putText(
            outlined,
            line1,
            (x, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        if line2:
            cv2.putText(
                outlined,
                line2,
                (x, max(30, y + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        cv2.putText(
            outlined,
            f"A={area:.0f}",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # draw lines last
    final_out = line_detector.draw_lines(outlined)

    cv2.imwrite(str(out_path), final_out)
    print(f"✅ Saved: {out_path}")

    if SHOW_WINDOW:
        cv2.imshow("Processed Image", final_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

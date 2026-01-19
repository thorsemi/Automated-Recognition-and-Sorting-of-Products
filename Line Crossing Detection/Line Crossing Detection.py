# yolo_video_overlay_with_lines.py
# Kører HELE videoen (ingen skip), tegner YOLO-klassenavne + hjælpelinjer,
# og gemmer som en jævn MP4 med original FPS/op­løsning.

from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO


# === KONFIG ===
MODEL_PATH = r"C:\Users\emilt\Desktop\YOLO\yolo_medium\Run2\y11m_medium_only_20251030\medium_only_20251030\weights\y11m_prod_tune_best.pt"
VIDEO_PATH = r"C:\Users\emilt\Desktop\Materiale_Final_Vieo\5 (Med i Video).avi"

CONF_THRES = 0.25
DRAW_BOXES = False          # True=tegn bokse + label. False=kun label ved center
SHOW_WINDOW = False         # True=vis live vindue (kan gøre det langsommere)
CHECK_LINE_HITS = True      # True=print koder når et objekts center er tæt på en linje
LINE_TOLERANCE = 10         # pixel-afstand for "rammer linje"

# Hjælpelinjer (samme som dit eksempel)
TOP_LINE_Y = 24
DISTANCE_BETWEEN = 779
BOTTOM_LINE_Y = TOP_LINE_Y + DISTANCE_BETWEEN
MIDDLE_X = 1500
MIDDLE_X_2 = 35

# Farver (BGR)
COLOR_TOP_BOTTOM = (0, 0, 255)   # rød
COLOR_MID = (0, 255, 0)          # grøn
COLOR_MID2 = (255, 0, 0)         # blå


def pick_color(label: str) -> tuple[int, int, int]:
    """Farvevalg for labels."""
    lab = label.lower()
    if "front" in lab:
        return (0, 255, 0)       # grøn
    if "back" in lab:
        return (0, 165, 255)     # orange
    return (0, 0, 255)           # rød


def draw_guides(img: np.ndarray) -> None:
    """Tegn faste hjælpelinjer på frame."""
    h, w = img.shape[:2]
    y1 = max(0, min(h - 1, TOP_LINE_Y))
    y2 = max(0, min(h - 1, BOTTOM_LINE_Y))
    x1 = max(0, min(w - 1, MIDDLE_X))
    x2 = max(0, min(w - 1, MIDDLE_X_2))

    # Vandrette linjer
    cv2.line(img, (0, y1), (w - 1, y1), COLOR_TOP_BOTTOM, 3)
    cv2.line(img, (0, y2), (w - 1, y2), COLOR_TOP_BOTTOM, 3)

    # Lodrette linjer
    cv2.line(img, (x1, y1), (x1, y2), COLOR_MID, 3)
    cv2.line(img, (x2, y1), (x2, y2), COLOR_MID2, 3)


def main() -> None:
    model = YOLO(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Kunne ikke åbne video: {VIDEO_PATH}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    in_path = Path(VIDEO_PATH)
    out_path = in_path.with_name(f"{in_path.stem}_lines_yolo.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    assert writer.isOpened(), "Kunne ikke åbne VideoWriter (tjek codec/sti)."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inferens pr. frame (ingen skip)
        res = model.predict(frame, device=device, conf=CONF_THRES, verbose=False)[0]
        boxes = res.boxes
        names = res.names

        # Tegn hjælpelinjer først (så labels tegnes ovenpå)
        draw_guides(frame)

        # Skriv kun klassenavn ved objektets center (som dit eksempel)
        if boxes is not None and boxes.xyxy is not None:
            for xyxy, cls in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = xyxy.astype(int)
                label = names[int(cls)]
                color = pick_color(label)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if DRAW_BOXES:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
                )

                # Valgfri "linje-ramt" test
                if CHECK_LINE_HITS:
                    # 3 = top-linje, 2 = bund-linje, 1 = midter x, 4 = ekstra midter x
                    if abs(cy - TOP_LINE_Y) <= LINE_TOLERANCE:
                        print(3)
                    elif abs(cy - BOTTOM_LINE_Y) <= LINE_TOLERANCE:
                        print(2)
                    elif abs(cx - MIDDLE_X) <= LINE_TOLERANCE:
                        print(1)
                    elif abs(cx - MIDDLE_X_2) <= LINE_TOLERANCE:
                        print(4)

        # Skriv ét output-frame per input-frame -> glat video
        writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("YOLO + Lines", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print(f"Gemte: {out_path} @ {fps:.2f} FPS ({width}x{height})")
    print("Note: OpenCV VideoWriter gemmer uden lyd.")


if __name__ == "__main__":
    main()

import cv2
from models.yolo_model import YOLOModel
from line_detection.line_detector import LineDetector

# === Paths ===

## Model: Nano 
##  YOLO_PATH = r"C:\Users\emilt\Desktop\YOLO\best.pt"

## Model: Small
## YOLO_PATH = r"C:\Users\emilt\Desktop\YOLO\best.pt"


YOLO_PATH = r"C:\Users\emilt\Desktop\Bilag\YOLO\medium\weights\y11m_prod_tune_best.pt"
IMAGE_PATH = r"C:\Users\emilt\Desktop\55_000027.png"


def main():
    # --- 1) Indlæs model og billede ---
    model = YOLOModel(YOLO_PATH)
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise FileNotFoundError(f"Kunne ikke finde billedet: {IMAGE_PATH}")

    print("🚀 Kører YOLO inference ...")
    result_frame, detections = model.predict(frame, return_boxes=True)

    # --- 2) Opsæt linjedetektion ---
    line_detector = LineDetector(top_line_y=24, distance_between=779, middle_x=1500, middle_x_2=35)

    # --- 3) Gennemgå alle objekter og tjek om de rammer en linje ---
    for (cx, cy) in detections:
        event = line_detector.check_collision(cx, cy)
        if event:
            print(f"⚡ Objekt ramte linje {event} (x={cx}, y={cy})")

    # --- 4) Tegn linjer og vis resultat ---
    final_frame = line_detector.draw_lines(result_frame)
    cv2.imshow("YOLO + Line Detection", final_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

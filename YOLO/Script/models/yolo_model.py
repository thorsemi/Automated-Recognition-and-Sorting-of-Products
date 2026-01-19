from ultralytics import YOLO
import cv2


class YOLOModel:
    """Wrapper class for YOLO model loading and inference."""

    def __init__(self, model_path):
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        print("✅ YOLO model loaded successfully")

    def predict(self, frame, return_boxes=False):
        """Runs YOLO inference and returns annotated frame (and optionally box centers)."""
        results = self.model(frame, verbose=False)
        annotated = results[0].plot()

        centers = []
        if return_boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            for box in boxes:
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers.append((cx, cy))

        return (annotated, centers) if return_boxes else annotated

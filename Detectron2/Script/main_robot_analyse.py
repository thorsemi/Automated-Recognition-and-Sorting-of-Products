import os
from pathlib import Path

import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# ====== KONFIG ======
CONFIG_YAML = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

MODEL_PATH = (
    r"C:\Users\emilt\Desktop\Detectron2\Train\output_detectron2\model_final.pth"
)

VIDEO_PATH_IN = r"C:\Users\emilt\Desktop\Materiale_Final_Vieo\5 (Med i Video).avi"
HEATMAP_BG_PATH = r"C:\Users\emilt\Desktop\55_000027.png"

VIDEO_PATH_OUT = r"C:\Users\emilt\Desktop\overlap_failure_heatmap_3types.avi"

NUM_CLASSES = 7
SCORE_THRESH = 0.5

# ====== OVERLAP-KLASSER ======
CLASS_LIGHT = 3     # 2-overlap
CLASS_MEDIUM = 4    # 3-overlap
CLASS_HEAVY = 5     # heavy overlap

# Farver/colormaps for hver klasse
COLORMAP_LIGHT = cv2.COLORMAP_OCEAN
COLORMAP_MEDIUM = cv2.COLORMAP_JET
COLORMAP_HEAVY = cv2.COLORMAP_HOT

HEATMAP_ALPHA = 0.55  # blending mod baggrund


def build_predictor() -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_YAML))
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)


def add_mask_to_heatmap(mask, heatmap):
    ys, xs = np.where(mask == 1)
    if len(xs) > 0:
        heatmap[ys, xs] += 1.0


def main() -> None:
    if not os.path.exists(VIDEO_PATH_IN):
        raise FileNotFoundError(f"Video ikke fundet: {VIDEO_PATH_IN}")

    bg = cv2.imread(HEATMAP_BG_PATH)
    if bg is None:
        raise RuntimeError("Baggrundsbilledet kunne ikke læses.")

    cap = cv2.VideoCapture(VIDEO_PATH_IN)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg = cv2.resize(bg, (width, height))

    out = cv2.VideoWriter(
        VIDEO_PATH_OUT,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )

    predictor = build_predictor()

    # Tre individuelle heatmaps
    heat_light = np.zeros((height, width), dtype=np.float32)
    heat_medium = np.zeros((height, width), dtype=np.float32)
    heat_heavy = np.zeros((height, width), dtype=np.float32)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"Frame {frame_idx}...")

        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")

        if len(instances) > 0:
            classes = instances.pred_classes.numpy()

            # === LIGHT (class 3) ===
            idx_light = np.where(classes == CLASS_LIGHT)[0]

            # === MEDIUM (class 4) ===
            idx_medium = np.where(classes == CLASS_MEDIUM)[0]

            # === HEAVY (class 5) ===
            idx_heavy = np.where(classes == CLASS_HEAVY)[0]

            # Brug masker hvis tilgængelig
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()

                for i in idx_light:
                    add_mask_to_heatmap(masks[i], heat_light)

                for i in idx_medium:
                    add_mask_to_heatmap(masks[i], heat_medium)

                for i in idx_heavy:
                    add_mask_to_heatmap(masks[i], heat_heavy)

            else:
                # fallback til boxes
                boxes = instances.pred_boxes.tensor.numpy().astype(int)
                for i in idx_light:
                    x1, y1, x2, y2 = boxes[i]
                    heat_light[y1:y2, x1:x2] += 1.0
                for i in idx_medium:
                    x1, y1, x2, y2 = boxes[i]
                    heat_medium[y1:y2, x1:x2] += 1.0
                for i in idx_heavy:
                    x1, y1, x2, y2 = boxes[i]
                    heat_heavy[y1:y2, x1:x2] += 1.0

        # ====== Normaliser + farvelæg ======
        def make_colormap(hm, cmap):
            if hm.max() > 0:
                hm_norm = (hm / hm.max()) * 255
            else:
                hm_norm = hm
            hm_uint8 = hm_norm.astype(np.uint8)
            return cv2.applyColorMap(hm_uint8, cmap)

        hm_light_color = make_colormap(heat_light, COLORMAP_LIGHT)
        hm_medium_color = make_colormap(heat_medium, COLORMAP_MEDIUM)
        hm_heavy_color = make_colormap(heat_heavy, COLORMAP_HEAVY)

        # Kombinér alle tre heatmaps
        combined = cv2.addWeighted(hm_light_color, 0.7, hm_medium_color, 0.7, 0)
        combined = cv2.addWeighted(combined, 1.0, hm_heavy_color, 0.8, 0)

        # Overlay på baggrund
        final_overlay = cv2.addWeighted(bg, 1 - HEATMAP_ALPHA, combined, HEATMAP_ALPHA, 0)

        out.write(final_overlay)

    cap.release()
    out.release()

    print("\n=========================================")
    print("Færdig! 3-klasse Overlap Heatmap gemt som:")
    print(VIDEO_PATH_OUT)
    print("=========================================\n")


if __name__ == "__main__":
    main()

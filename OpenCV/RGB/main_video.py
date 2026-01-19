# main_video.py
from pathlib import Path
import cv2
import numpy as np

from detection import detect_leaflets, SINGLE_AREA_REF
from classification import classify_front_back

# >>> RET DENNE TIL DIN VIDEO <<<
VIDEO_PATH = r"C:\Users\emilt\Desktop\Materiale_Final_Vieo\5 (Med i Video).avi"

SHOW_WINDOW = True    # show result in a window
SHOW_MASK = False     # (optional) show detection mask

# Anything above this area is treated as overlap (fallback)
AREA_OVERLAP_THRESHOLD = 16000.0


def main() -> None:
    video_path = Path(VIDEO_PATH)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    # --- output video info ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    out_path = video_path.with_name(video_path.stem + "_processed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    print(f"▶️  Reading:  {VIDEO_PATH}")
    print(f"💾 Writing:  {out_path}")
    print(f"   FPS={fps:.2f}, size={width}x{height}")

    frame_idx = 0
    delay = max(int(1000 / fps), 1)

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_idx += 1

        # --- detection on BGR frame ---
        boxes, rects, mask_det, valid_contours, overlap_contours = detect_leaflets(
            frame_bgr
        )

        outlined = frame_bgr.copy()

        # compute base area (median single or fallback ref)
        single_areas = [cv2.contourArea(c) for c in valid_contours]
        if single_areas:
            base_area = float(np.median(single_areas))
        else:
            base_area = SINGLE_AREA_REF

        # --- single leaflets + front/back + area-based overlap fallback ---
        for idx, (box, rect, num_corners) in enumerate(boxes):
            cnt = valid_contours[idx]
            area = cv2.contourArea(cnt)

            # Fallback: very large area → treat as overlap and skip front/back
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

            # normal single leaflet: draw green + classify front/back
            cv2.drawContours(outlined, [box], 0, (0, 255, 0), 2)

            label, ratio = classify_front_back(frame_bgr, box)
            cx, cy = rect[0]

            text1 = f"{label} ({ratio:.2f})"
            text2 = f"A={area:.0f}"

            # Front/Back in blue
            cv2.putText(
                outlined,
                text1,
                (int(cx), int(cy) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            # Area in green
            cv2.putText(
                outlined,
                text2,
                (int(cx), int(cy) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # --- geometric overlap candidates from detection ---
        for ov in overlap_contours:
            cnt = ov["cnt"]
            area = ov["area"]
            reason = ov["reason"]

            rect = cv2.minAreaRect(cnt)
            box2 = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(outlined, [box2], 0, (0, 0, 255), 3)

            x, y, w, h = cv2.boundingRect(cnt)

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
                (x, max(15, y - 10)),
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
                f"A={area:.0f} [{reason}]",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # write frame
        writer.write(outlined)

        # show live
        if SHOW_WINDOW:
            cv2.imshow("Processed", outlined)
            if SHOW_MASK:
                cv2.imshow("Detection mask", mask_det)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q") or key == 27:
                break

        if frame_idx % 100 == 0:
            print(f"... {frame_idx} frames processed")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("✅ Done. Saved:", out_path)


if __name__ == "__main__":
    main()

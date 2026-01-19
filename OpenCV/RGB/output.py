import cv2
import os
import datetime


# Function to handle recording start/stop for video output
def save_recording(recording, out, frame, save_path):
    if not recording:
        os.makedirs(save_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = os.path.join(
            save_path,
            datetime.datetime.now().strftime("recording_%Y%m%d_%H%M%S.avi")
        )
        h, w, _ = frame.shape
        out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
        recording = True
        print(f"[INFO] Recording started: {filename}")
    else:
        recording = False
        if out is not None:
            out.release()
            out = None
        print("[INFO] Recording ended")

    return recording, out

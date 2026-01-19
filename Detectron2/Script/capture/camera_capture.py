import cv2
import numpy as np
from websocket import create_connection


class CameraCapture:
    """Handles real-time frame capture via WebSocket."""

    def __init__(self, url="ws://192.168.1.120:8888/ws"):
        try:
            self.ws = create_connection(url)
            print(f"✅ Connected to camera at {url}")
        except Exception as e:
            print("❌ Failed to connect to server:", e)
            self.ws = None

    def get_frame(self):
        """Receives a single frame from the WebSocket stream."""
        if not self.ws:
            return None

        try:
            frame_bytes = self.ws.recv()
            img_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print("⚠️ Frame reception error:", e)
            return None

    def close(self):
        """Closes the WebSocket connection."""
        if self.ws:
            self.ws.close()
            print("🔌 WebSocket connection closed")

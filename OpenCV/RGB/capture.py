import cv2
import numpy as np
from websocket import create_connection

# Class to handle camera capture via WebSocket
class CameraCapture:
    def __init__(self, url="ws://192.168.1.120:8888/ws"):
        """Connects to the camera via WebSocket"""
        try:
            self.ws = create_connection(url)
            print(f"Connected to server {url}")
        except Exception as e:
            print("Cant connect to server:", e)
            self.ws = None

    def get_frame(self):
        """Downloads a frame from the camera"""
        if self.ws is None:
            return None

        try:
            frame_bytes = self.ws.recv()
            img_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print("Error:", e)
            return None

    def close(self):
        """Closes the WebSocket connection"""
        if self.ws:
            self.ws.close()
            print("WebSocket connection ended")

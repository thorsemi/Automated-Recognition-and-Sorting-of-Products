import cv2
import numpy as np


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
        cv2.line(frame_display, (self.middle_x, self.top_line_y), (self.middle_x, self.bottom_line_y), (0, 255, 0), 3)
        cv2.line(frame_display, (self.middle_x_2, self.top_line_y), (self.middle_x_2, self.bottom_line_y), (255, 0, 0), 3)

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

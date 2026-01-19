import cv2


def show_frame(window_name, frame):
    """Displays a frame in an OpenCV window."""
    cv2.imshow(window_name, frame)


def write_frame(video_writer, frame):
    """Writes a frame to the output video file."""
    if video_writer is not None:
        video_writer.write(frame)

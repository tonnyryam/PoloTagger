import cv2
import numpy as np

def load_detector():
    """
    Simulate loading a model for detecting cap numbers.
    Replace this with real model loading when available.
    """
    print("[cap_number] Simulated detector loaded.")
    return None

def identify_numbers_in_frame(frame, model=None):
    """
    Simulate identifying cap numbers by team in a frame.

    Args:
        frame (np.ndarray): BGR image (H x W x 3)
        model: Optional detection model

    Returns:
        dict: {"W": [int, ...], "D": [int, ...]}
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Simulated color ranges for white and dark caps
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))  # white
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))     # black/navy

    # Count blobs and simulate fake numbers (e.g., cap #s 1–6)
    w_count = cv2.countNonZero(white_mask) // 5000
    d_count = cv2.countNonZero(dark_mask) // 5000

    white_caps = list(range(1, w_count + 1))
    dark_caps = list(range(1, d_count + 1))

    return {"W": white_caps, "D": dark_caps}
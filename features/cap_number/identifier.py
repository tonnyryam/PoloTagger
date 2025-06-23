import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# Load YOLO model once (we use a custom trained one or pretrained on "person"/"head")
cap_detector = None

def load_detector(model_path="yolov8n.pt"):
    global cap_detector
    cap_detector = YOLO(model_path)  # Replace with custom trained model path if available
    print("[cap_number] YOLO detector loaded.")
    return cap_detector

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def recognize_number(crop):
    ocr_ready = preprocess_for_ocr(crop)
    config = "--psm 8 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(ocr_ready, config=config)
    text = ''.join(filter(str.isdigit, text))
    return int(text) if text.isdigit() else None

def identify_numbers_in_frame(frame, model=None):
    if model is None:
        model = cap_detector
    if model is None:
        raise ValueError("Model not loaded. Call load_detector() first.")

    results = model(frame)[0]
    caps_by_team = {"W": [], "D": []}

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        number = recognize_number(crop)
        if number is None:
            continue

        # Simple heuristic based on brightness to split W/D
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_v = hsv[..., 2].mean()
        team = "W" if avg_v > 128 else "D"
        caps_by_team[team].append(number)

    return caps_by_team
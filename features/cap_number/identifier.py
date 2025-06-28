import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as T

cap_detector = None
classifier = None

transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def load_detector(yolo_path="yolov8n.pt", digit_model_path=None):
    global cap_detector, classifier
    cap_detector = YOLO(yolo_path)
    print("[cap_number] YOLO detector loaded.")

    if digit_model_path:
        classifier = DigitClassifier()
        classifier.load_state_dict(torch.load(digit_model_path, map_location="cpu"))
        classifier.eval()
        print("[cap_number] Digit classifier loaded.")
    else:
        print("[cap_number] No digit classifier loaded.")
    return cap_detector

def recognize_number(crop):
    if classifier is None:
        return None
    try:
        x = transform(crop).unsqueeze(0)
        with torch.no_grad():
            out = classifier(x)
            pred = torch.argmax(out, dim=1).item()
        return pred
    except:
        return None

def identify_numbers_in_frame(frame, model=None):
    if model is None:
        model = cap_detector
    if model is None:
        raise ValueError("Detector not loaded")

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        number = recognize_number(crop)
        if number is None:
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_v = hsv[..., 2].mean()

        detections.append({"number": number, "avg_v": avg_v})

    # Sort by brightness and split
    sorted_caps = sorted(detections, key=lambda d: d["avg_v"])
    split = len(sorted_caps) // 2
    team_D = sorted_caps[:split]
    team_W = sorted_caps[split:]

    return {
        "W": [d["number"] for d in team_W],
        "D": [d["number"] for d in team_D],
        "meta": sorted_caps
    }

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as T
import urllib.request
import os
from pathlib import Path

cap_detector = None
classifier = None

transform = T.Compose(
    [
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((28, 28)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ]
)


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
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def download_model_if_needed(url, local_path):
    """Download model if it doesn't exist locally"""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if not local_path.exists():
        print(f"[INFO] Downloading digit model from {url}")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"[INFO] Model downloaded to {local_path}")
            return str(local_path)
        except Exception as e:
            print(f"[WARNING] Download failed: {e}")
            return None
    return str(local_path)


def load_detector(yolo_path="yolov8n.pt", digit_model_path=None):
    global cap_detector, classifier
    cap_detector = YOLO(yolo_path)
    print("[cap_number] YOLO detector loaded.")

    if digit_model_path:
        classifier = DigitClassifier()

        # Handle auto-download for mnist-onnx
        if digit_model_path == "mnist-onnx":
            # Try multiple sources for MNIST models
            model_sources = [
                {
                    "url": "https://github.com/pytorch/examples/raw/main/mnist/mnist_cnn.pt",
                    "path": "models/mnist_digit_classifier.pt",
                },
                {
                    "url": "https://download.pytorch.org/models/mnist_cnn.pt",
                    "path": "models/mnist_digit_classifier_alt.pt",
                },
            ]

            model_loaded = False
            for source in model_sources:
                downloaded_path = download_model_if_needed(
                    source["url"], source["path"]
                )
                if downloaded_path and os.path.exists(downloaded_path):
                    try:
                        classifier.load_state_dict(
                            torch.load(downloaded_path, map_location="cpu")
                        )
                        classifier.eval()
                        print(
                            f"[cap_number] Digit classifier loaded from {downloaded_path}"
                        )
                        model_loaded = True
                        break
                    except Exception as e:
                        print(
                            f"[WARNING] Could not load model from {downloaded_path}: {e}"
                        )
                        continue

            if not model_loaded:
                print("[WARNING] Could not download any digit models")
                print("[cap_number] Using untrained digit classifier")
                classifier.eval()

        else:
            # Load from specific file path
            if os.path.exists(digit_model_path):
                classifier.load_state_dict(
                    torch.load(digit_model_path, map_location="cpu")
                )
                classifier.eval()
                print(f"[cap_number] Digit classifier loaded from {digit_model_path}")
            else:
                print(f"[WARNING] Digit model file not found: {digit_model_path}")
                print("[cap_number] Using untrained digit classifier")
                classifier.eval()
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
        "meta": sorted_caps,
    }

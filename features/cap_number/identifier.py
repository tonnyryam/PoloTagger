import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as T
import urllib.request
from pathlib import Path

cap_detector = None
classifier   = None

transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
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
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def ensure_pretrained_models():
    """
    Download YOLOv8n and MNIST digit CNN weights into this feature's models/ directory.
    Returns paths to the downloaded files.
    """
    base_dir = Path(__file__).parent
    model_dir = base_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    # YOLOv8n weights
    yolo_path = model_dir / 'yolov8n.pt'
    if not yolo_path.exists():
        yolo_url = 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt'
        urllib.request.urlretrieve(yolo_url, yolo_path)

    # MNIST CNN weights (PyTorch example)
    digit_path = model_dir / 'mnist_digit_classifier.pt'
    if not digit_path.exists():
        mnist_url = 'https://github.com/pytorch/examples/raw/main/mnist/mnist_cnn.pt'
        urllib.request.urlretrieve(mnist_url, digit_path)

    return str(yolo_path), str(digit_path)


def load_detector():
    """
    Initialize YOLO and digit classifier models, downloading weights if needed.
    """
    global cap_detector, classifier

    yolo_model_path, digit_model_path = ensure_pretrained_models()

    # Load YOLO detector
    cap_detector = YOLO(yolo_model_path)
    print(f"[cap_number] YOLO detector loaded from {yolo_model_path}")

    # Load digit classifier
    classifier = DigitClassifier()
    try:
        state = torch.load(digit_model_path, map_location='cpu')
        classifier.load_state_dict(state)
        classifier.eval()
        print(f"[cap_number] Digit classifier loaded from {digit_model_path}")
    except Exception as e:
        print(f"[cap_number] Failed to load digit classifier: {e}")

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
    except Exception:
        return None


def identify_numbers_in_frame(frame, model=None):
    if model is None:
        model = cap_detector
    if model is None:
        raise RuntimeError("Detector not initialized")

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
        detections.append({'number': number, 'avg_v': avg_v})

    sorted_caps = sorted(detections, key=lambda d: d['avg_v'])
    split = len(sorted_caps) // 2
    return {
        'W': [d['number'] for d in sorted_caps[split:]],
        'D': [d['number'] for d in sorted_caps[:split]],
        'meta': sorted_caps,
    }

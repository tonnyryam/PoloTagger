# features/cap_number/identifier.py
"""
This module initializes and provides functions for detecting numbered caps in video frames.
It checks for pretrained YOLOv8n and a PyTorch‐CNN MNIST model in its own models/ directory,
and only downloads them if they are missing.
"""

import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
import urllib.request

# Globals for detector and digit classifier
cap_detector = None
classifier = None

# Transform for digit classifier input
digit_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((28, 28)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ]
)


class DigitClassifier(nn.Module):
    """Simple CNN for MNIST digit recognition."""

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


def download_if_missing(url: str, dest: Path):
    """Download a file from URL to dest if not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[cap_number] Downloading {url} to {dest.name}")
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f"[cap_number] Download complete: {dest.name}")
        except Exception as e:
            print(f"[cap_number] WARNING: failed to download {url}: {e}")


def ensure_mnist_model():
    """
    Ensure that the MNIST CNN weights are available locally.
    Returns path to the CNN weights.
    """
    base_dir = Path(__file__).parent
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    mnist_url = "https://github.com/pytorch/examples/raw/main/mnist/mnist_cnn.pt"
    mnist_path = model_dir / "mnist_digit_classifier.pt"
    download_if_missing(mnist_url, mnist_path)
    return str(mnist_path)


def load_detector():
    """
    Initialize the YOLOv8n detector and MNIST CNN digit classifier.
    Downloads weights into features/cap_number/models/ if needed.
    Returns the YOLO detector instance.
    """
    global cap_detector, classifier
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) YOLOv8n detector
    yolo_local = model_dir / "yolov8n.pt"
    if yolo_local.exists():
        cap_detector = YOLO(str(yolo_local))
        print(f"[cap_number] YOLO detector loaded from local {yolo_local.name}")
    else:
        cap_detector = YOLO(
            "yolov8n"
        )  # auto-downloads and caches under ~/.cache/ultralytics
        print("[cap_number] YOLO detector auto-downloaded via Ultralytics")

    # 2) MNIST CNN classifier
    mnist_model_path = ensure_mnist_model()
    classifier = DigitClassifier()
    try:
        state = torch.load(mnist_model_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        classifier.load_state_dict(state)
        classifier.eval()
        print(
            f"[cap_number] Digit classifier loaded from {Path(mnist_model_path).name}"
        )
    except Exception as e:
        print(f"[cap_number] WARNING: failed to load digit classifier: {e}")
        classifier = None

    return cap_detector


def identify_numbers_in_frame(frame, model=None):
    """
    Detects caps in the frame using YOLO and reads digits using the CNN.
    Returns a dict with keys 'W', 'D', and 'meta'.
    """
    if model is None:
        model = cap_detector
    if model is None:
        raise RuntimeError("cap_number: YOLO detector not initialized")

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        number = None
        if classifier:
            try:
                x = digit_transform(crop).unsqueeze(0)
                with torch.no_grad():
                    number = int(classifier(x).argmax())
            except Exception:
                number = None
        if number is None:
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_v = float(hsv[..., 2].mean())
        detections.append({"number": number, "avg_v": avg_v})

    sorted_caps = sorted(detections, key=lambda d: d["avg_v"])
    mid = len(sorted_caps) // 2
    return {
        "W": [d["number"] for d in sorted_caps[mid:]],
        "D": [d["number"] for d in sorted_caps[:mid]],
        "meta": sorted_caps,
    }

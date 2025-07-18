# features/cap_number/identifier.py

"""
This module initializes and provides functions for detecting numbered caps in video frames.
It checks for pretrained YOLOv8n and a Torch .pt MNIST model in its own models/ directory.
"""

import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torchvision.transforms as T

# Globals for detector and digit classifier
cap_detector = None
classifier = None


# ─── MNIST CNN Definition ────────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


# ─── Preprocessing for digit classifier ───────────────────────
digit_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((28, 28)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ]
)


def load_detector():
    """
    Load the YOLOv8n detector and a TorchScript-like MNIST digit classifier (.pt).
    Expects these files under features/cap_number/models/:
      - yolov8n.pt
      - mnist_cnn.pt

    Returns:
        cap_detector (YOLO): the loaded object detector
    """
    global cap_detector, classifier

    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) YOLOv8n detector ─────────────────────────────────────
    yolo_path = model_dir / "yolov8n.pt"
    if yolo_path.exists():
        cap_detector = YOLO(str(yolo_path))
        print(f"[cap_number] YOLO detector loaded from local {yolo_path.name}")
    else:
        cap_detector = YOLO("yolov8n")
        print("[cap_number] YOLO detector auto-downloaded via Ultralytics")

    # ── 2) Torch .pt digit classifier ───────────────────────────
    pt_path = model_dir / "mnist_cnn.pt"
    if pt_path.exists():
        # Instantiate architecture and load state_dict
        net = Net().to("cpu")
        state = torch.load(str(pt_path), map_location="cpu")
        if isinstance(state, dict):
            net.load_state_dict(state)
        else:
            # if the file itself is a scripted module
            net = state
        net.eval()

        def _classifier(image_tensor: torch.Tensor):
            # image_tensor: [N,1,28,28]
            logits = net(image_tensor)
            preds = logits.argmax(1)
            if preds.shape[0] == 1:
                return int(preds.item())
            return [int(p.item()) for p in preds]

        classifier = _classifier
        print(f"[cap_number] Digit classifier loaded from {pt_path.name}")
    else:
        print(f"[cap_number] WARNING: Torch .pt model not found at {pt_path}")
        classifier = None

    return cap_detector


def identify_numbers_in_frame(frame, model=None):
    """
    Detects caps in the frame using YOLO and reads digits using the classifier.
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
                number = classifier(x)
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

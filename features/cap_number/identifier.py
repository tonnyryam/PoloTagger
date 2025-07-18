# features/cap_number/identifier.py

"""
This module initializes and provides functions for detecting numbered caps in video frames.
It checks for pretrained YOLOv8n and an ONNX‐based MNIST model in its own models/ directory.
"""

import cv2
from ultralytics import YOLO
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path
import torchvision.transforms as T

# Globals for detector and digit classifier
cap_detector = None
classifier = None

# Transform pipeline for digit classifier input
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
    Initialize the YOLOv8n detector and an ONNX‐based MNIST digit classifier.
    Expects these files in features/cap_number/models/:
      - yolov8n.pt
      - mnist-8.onnx

    Returns:
        cap_detector (YOLO): the loaded object detector
    """
    global cap_detector, classifier

    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) YOLOv8n detector ─────────────────────────────────────
    yolo_local = model_dir / "yolov8n.pt"
    if yolo_local.exists():
        cap_detector = YOLO(str(yolo_local))
        print(f"[cap_number] YOLO detector loaded from local {yolo_local.name}")
    else:
        cap_detector = YOLO("yolov8n")  # auto-download via Ultralytics cache
        print("[cap_number] YOLO detector auto-downloaded via Ultralytics")

    # ── 2) ONNX digit classifier ─────────────────────────────────
    onnx_file = model_dir / "mnist-8.onnx"
    if not onnx_file.exists():
        print(f"[cap_number] WARNING: ONNX model not found at {onnx_file}")
        classifier = None
    else:
        session = ort.InferenceSession(
            str(onnx_file), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name

        def _classifier(image_tensor: torch.Tensor):
            """
            image_tensor: torch.Tensor of shape [N,1,28,28], dtype float32
            returns: int if N==1 else List[int]
            """
            np_in = image_tensor.cpu().numpy().astype(np.float32)
            outputs = session.run(None, {input_name: np_in})
            preds = np.argmax(outputs[0], axis=1)
            if preds.shape[0] == 1:
                return int(preds[0])
            return [int(p) for p in preds]

        classifier = _classifier
        print(f"[cap_number] Digit classifier loaded from {onnx_file.name}")

    return cap_detector


def identify_numbers_in_frame(frame, model=None):
    """
    Detects caps in the frame using YOLO and reads digits using the ONNX classifier.
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

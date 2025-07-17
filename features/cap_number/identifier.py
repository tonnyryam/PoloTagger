# features/cap_number/identifier.py
"""
This module initializes and provides functions for detecting numbered caps in video frames.
It checks for pretrained YOLOv8n and MNIST ONNX models in its own models/ directory
and only downloads them if they are missing.
"""

import cv2
from ultralytics import YOLO
import onnxruntime as ort
import torchvision.transforms as T
from pathlib import Path
import urllib.request

# Globals for detector and digit ONNX session
cap_detector = None
digit_session = None

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


def download_if_missing(url: str, dest: Path):
    """Download a file from URL to dest if not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[cap_number] Downloading {url} to {dest}")
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f"[cap_number] Download complete: {dest.name}")
        except Exception as e:
            print(f"[cap_number] WARNING: failed to download {url}: {e}")


def ensure_mnist_onnx_model():
    """
    Ensure that the MNIST ONNX model is available locally.
    Returns path to the ONNX model file.
    """
    base_dir = Path(__file__).parent
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Use ONNX Model Zoo path
    onnx_url = (
        "https://github.com/onnx/models/raw/main/validated/vision/classification/"
        "mnist/model/mnist-8.onnx"
    )
    onnx_path = model_dir / "mnist-8.onnx"
    download_if_missing(onnx_url, onnx_path)
    return str(onnx_path)


def load_detector():
    """
    Initialize the YOLOv8n detector and MNIST ONNX digit classifier.
    Downloads weights into features/cap_number/models/ if needed.
    Returns the YOLO detector instance.
    """
    global cap_detector, digit_session
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) YOLOv8n detector
    yolo_local = model_dir / "yolov8n.pt"
    if yolo_local.exists():
        cap_detector = YOLO(str(yolo_local))
        print(f"[cap_number] YOLO detector loaded from local {yolo_local.name}")
    else:
        cap_detector = YOLO("yolov8n")  # auto-downloads to cache
        print(f"[cap_number] YOLO detector loaded (auto-downloaded)")

    # 2) MNIST ONNX classifier
    onnx_model_path = ensure_mnist_onnx_model()
    try:
        digit_session = ort.InferenceSession(onnx_model_path)
        print(f"[cap_number] Digit classifier loaded from {Path(onnx_model_path).name}")
    except Exception as e:
        print(f"[cap_number] WARNING: failed to load ONNX digit model: {e}")
        digit_session = None

    return cap_detector


def identify_numbers_in_frame(frame, model=None):
    """
    Detects caps in the frame using YOLO and reads digits using the ONNX classifier.
    Returns a dict: {'W': [...], 'D': [...], 'meta': [...]}.
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
        if digit_session:
            try:
                x = digit_transform(crop).unsqueeze(0).numpy()
                out = digit_session.run(None, {digit_session.get_inputs()[0].name: x})[
                    0
                ]
                number = int(out.argmax())
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

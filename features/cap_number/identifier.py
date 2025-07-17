# features/cap_number/identifier.py
"""
This module initializes and provides functions for detecting numbered caps in video frames.
It downloads and loads pretrained YOLOv8n and MNIST digit-recognition models into its own models/ directory.
"""

import cv2
from ultralytics import YOLO
import onnxruntime as ort
import torchvision.transforms as T
from pathlib import Path
import urllib.request

# Globals for detector and digit session
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


class ONNXDigitClassifier:
    """Wraps an ONNX Runtime session for MNIST classification."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        # assume input name is first input\
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, crop):
        # Preprocess crop and run inference
        x = digit_transform(crop).unsqueeze(0).numpy()
        outputs = self.session.run(None, {self.input_name: x})[0]
        return int(outputs.argmax())


def download_if_missing(url: str, dest: Path):
    """Download a file from URL to dest if not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[cap_number] Downloading {url} to {dest}")
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as e:
            print(f"[cap_number] WARNING: failed to download {url}: {e}")


def ensure_pretrained_models():
    """
    Ensure that both YOLOv8n and MNIST ONNX models are available locally.
    Returns paths for the digit ONNX model; YOLO weights are managed by the Ultralytics library.
    """
    base_dir = Path(__file__).parent
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download MNIST ONNX model if missing
    onnx_url = (
        "https://raw.githubusercontent.com/onnx/models/main/vision/classification/"
        "mnist/model/mnist-8.onnx"
    )
    onnx_path = model_dir / "mnist-8.onnx"
    download_if_missing(onnx_url, onnx_path)

    return str(onnx_path)


def load_detector():
    """
    Initialize the YOLO detector and MNIST digit classifier.
    Downloads weights into features/cap_number/models/ if needed.
    Returns the YOLO detector.
    """
    global cap_detector, digit_session

    # 1) Instantiate YOLOv8n detector (auto-download by library)
    try:
        cap_detector = YOLO("yolov8n")
        print(f"[cap_number] YOLO detector loaded (yolov8n)")
    except Exception as e:
        print(f"[cap_number] ERROR initializing YOLO detector: {e}")
        cap_detector = None

    # 2) Ensure and load MNIST ONNX classifier
    onnx_model_path = ensure_pretrained_models()
    try:
        digit_session = ONNXDigitClassifier(onnx_model_path)
        print(f"[cap_number] Digit classifier loaded from {Path(onnx_model_path).name}")
    except Exception as e:
        print(f"[cap_number] WARNING loading digit classifier: {e}")
        digit_session = None

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

    # Run YOLO detection
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        number = None
        if digit_session:
            try:
                number = digit_session(crop)
            except:
                number = None
        if number is None:
            continue

        # Compute brightness for sorting
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_v = float(hsv[..., 2].mean())
        detections.append({"number": number, "avg_v": avg_v})

    # Sort and split
    det_sorted = sorted(detections, key=lambda d: d["avg_v"])
    mid = len(det_sorted) // 2
    return {
        "W": [d["number"] for d in det_sorted[mid:]],
        "D": [d["number"] for d in det_sorted[:mid]],
        "meta": det_sorted,
    }

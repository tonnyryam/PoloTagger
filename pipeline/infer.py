import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
import torchvision.transforms as T
import cv2
import numpy as np
from features.cap_number.identifier import identify_numbers_in_frame, load_detector
from preprocess import parse_sportscode_xml
from export_predictions_to_xml import export_predictions_to_xml
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix

label_list = [
    "W Possession", "W Turn Over", "D Possession", "D CA", "D Turn Over",
    "Referee", "W CA", "W DEXC", "W 6/5", "D Goals", "D Shots", "D 5/6",
    "W FCO", "W AG", "W New 20", "D Center", "W Shots", "W Center",
    "D DEXC", "D 6/5", "W 5/6", "D Penalty", "W Penalty", "W Goals",
    "D AG", "D FCO", "W Time Out"
]

def export_predictions_to_xml(predictions, output_file="hudl_tags.xml", duration=5.0):
    """
    predictions: list of (timestamp, [labels], cap_numbers) tuples
    """
    root = ET.Element("Tags")

    for timestamp, labels, cap_numbers in predictions:
        for label in labels:
            event = ET.SubElement(root, "Event")
            start = ET.SubElement(event, "Start")
            end = ET.SubElement(event, "End")
            tag = ET.SubElement(event, "Label")

            start.text = f"{timestamp:.2f}"
            end.text = f"{(timestamp + duration):.2f}"
            tag.text = label

            if cap_numbers:
                caps = ET.SubElement(event, "Caps")
                w_team = ET.SubElement(caps, "W")
                d_team = ET.SubElement(caps, "D")
                w_team.text = ", ".join(str(c) for c in cap_numbers.get("W", []))
                d_team.text = ", ".join(str(c) for c in cap_numbers.get("D", []))

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Saved XML with cap numbers to {output_file}")

def compute_metrics(predictions, ground_truth, label_list, time_tolerance=2.0):
    """
    predictions: [(timestamp, [label, ...]), ...]
    ground_truth: [(timestamp, [label, ...]), ...]
    """
    num_labels = len(label_list)
    label_to_idx = {label: i for i, label in enumerate(label_list)}

    def binarize(events):
        bin_vector = [0] * num_labels
        for label in events:
            if label in label_to_idx:
                bin_vector[label_to_idx[label]] = 1
        return bin_vector

    # Match predictions to ground truth clips using time tolerance
    y_true = []
    y_pred = []

    gt_used = set()
    for pt, pred_labels in predictions:
        best_match = None
        for i, (gt_time, gt_labels) in enumerate(ground_truth):
            if i in gt_used:
                continue
            if abs(gt_time - pt) <= time_tolerance:
                best_match = i
                break

        if best_match is not None:
            y_true.append(binarize(ground_truth[best_match][1]))
            gt_used.add(best_match)
        else:
            y_true.append([0] * num_labels)

        y_pred.append(binarize(pred_labels))

    # Add unused ground truth as false negatives
    for i, (gt_time, gt_labels) in enumerate(ground_truth):
        if i not in gt_used:
            y_true.append(binarize(gt_labels))
            y_pred.append([0] * num_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": multilabel_confusion_matrix(y_true, y_pred)
    }

    return metrics

def load_model(model_path, num_classes, device):
    model = r3d_18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_video(video_path, clip_len=5, fps=30, stride=2.5, transform=None):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    transform = transform or T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    clip_frames = int(clip_len * fps)
    step_frames = int(stride * fps)

    clips = []
    for start_f in range(0, total_frames - clip_frames + 1, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames = []
        for _ in range(clip_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)
        if len(frames) == clip_frames:
            clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]
            clips.append((start_f / original_fps, clip_tensor))
    cap.release()
    return clips

def predict_events(model, clips, label_list, device, cap_model=None, threshold=0.5):
    predictions = []
    for timestamp, clip in clips:
        clip = clip.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(clip)
            probs = torch.sigmoid(logits)[0]
            labels = [label_list[i] for i, p in enumerate(probs) if p > threshold]

        last_frame = clip[0, :, -1].permute(1, 2, 0).detach().cpu().numpy()
        last_frame = (last_frame * 255).astype("uint8")
        caps = identify_numbers_in_frame(last_frame, cap_model)
        print(f"{timestamp:.1f}s: {', '.join(labels)}")
        print(f"[cap_number] W caps: {caps['W']}, D caps: {caps['D']}")

        if labels:
            predictions.append((timestamp, labels, caps))
    return predictions

def run(video_path, model_path, ground_truth_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = load_model(model_path, len(label_list), device)
    cap_model = load_detector()
    clips = preprocess_video(video_path)
    predictions = predict_events(model, clips, label_list, device, cap_model)

    if ground_truth_path:
        ground_truth = parse_sportscode_xml(ground_truth_path)
        metrics = compute_metrics(predictions, ground_truth, label_list)
        print("\\nEVALUATION METRICS:")
        print("Precision:", metrics["precision"])
        print("Recall:", metrics["recall"])
        print("F1 Score:", metrics["f1"])

    export_predictions_to_xml(predictions, "hudl_tags.xml")
    print("Exported HUDL-compatible tags to hudl_tags.xml")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python infer.py <video_path> <model_path> [<ground_truth_xml>]")
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
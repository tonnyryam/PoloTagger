import torch
import torch.nn.functional as F
from torchvision.models.video import r3d_18
import torchvision.transforms as T
import cv2
import numpy as np
import os

def load_trained_model(model_path, label_list, device):
    model = r3d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_list))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_video(video_path, clip_len=5, fps=30, transform=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / original_fps

    stride = 2.5  # seconds between clips
    timestamps = np.arange(0, duration - clip_len, stride)

    transform = transform if transform else T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    clips = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        frames = []
        for _ in range(int(clip_len * fps)):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)
        if len(frames) == int(clip_len * fps):
            clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]
            clips.append((t, clip_tensor))
    cap.release()
    return clips

def run_inference(model, clips, label_list, device, threshold=0.5):
    predictions = []
    for timestamp, clip in clips:
        clip = clip.unsqueeze(0).to(device)  # add batch dim
        with torch.no_grad():
            output = model(clip)  # [1, num_labels]
            probs = torch.sigmoid(output)[0]  # [num_labels]
            labels = [label_list[i] for i, p in enumerate(probs) if p > threshold]
            if labels:
                predictions.append((timestamp, labels))
    return predictions

# Example usage
def infer_on_video(video_path, model_path, label_list, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = load_trained_model(model_path, label_list, device)
    clips = preprocess_video(video_path, clip_len=5, fps=30)
    results = run_inference(model, clips, label_list, device)
    for timestamp, events in results:
        print(f"{timestamp:.2f}s: {', '.join(events)}")
    return results

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import numpy as np
import cv2

class VideoClipDataset(Dataset):
    def __init__(self, metadata_csv, label_list, clip_len=5, fps=30, transform=None, permute=True):
        self.metadata = pd.read_csv(metadata_csv)
        self.label_list = label_list
        self.label_to_idx = {label: idx for idx, label in enumerate(label_list)}
        self.clip_len = clip_len
        self.fps = fps
        self.num_frames = int(clip_len * fps)
        self.permute = permute
        self.transform = transform if transform else T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        clip_path = row['clip_path']
        label_str = row['labels']

        frames = self.load_video_frames(clip_path)

        label_vector = torch.zeros(len(self.label_list))
        for label in str(label_str).split(";"):
            label = label.strip()
            if label in self.label_to_idx:
                label_vector[self.label_to_idx[label]] = 1

        if self.permute:
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]

        return frames, label_vector

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total - 1, self.num_frames).astype(int)

        frames = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_idxs:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
            i += 1
        cap.release()

        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return torch.stack(frames)  # [T, C, H, W]
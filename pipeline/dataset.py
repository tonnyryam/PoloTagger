import os
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pipeline.augmentations import VideoAugmentation

class VideoClipDataset(Dataset):
    def __init__(self, metadata_csv, label_list, clip_len=5, fps=30, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.label_list = label_list
        self.clip_len = clip_len
        self.fps = fps
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        video_path = row["video_path"]
        start_time = float(row["start_time"])
        label_vector = self._get_label_vector(row["labels"])

        frames = self._load_clip(video_path, start_time)

        if self.transform:
            frames = self.transform(frames)

        clip_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
        label_tensor = torch.tensor(label_vector, dtype=torch.float32)

        return clip_tensor, label_tensor

    def _load_clip(self, video_path, start_time):
        cap = cv2.VideoCapture(video_path)
        start_frame = int(start_time * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.clip_len * self.fps):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)

        cap.release()

        if len(frames) < self.clip_len * self.fps:
            pad = self.clip_len * self.fps - len(frames)
            frames.extend([frames[-1]] * pad)

        return np.stack(frames, axis=0)

    def _get_label_vector(self, label_str):
        label_vector = [0] * len(self.label_list)
        for label in label_str.split(","):
            label = label.strip()
            if label in self.label_list:
                label_vector[self.label_list.index(label)] = 1
        return label_vector

def load_train_val_datasets(csv_path, label_list, val_ratio=0.2):
    full_ds = VideoClipDataset(csv_path, label_list)

    # Split
    val_len = int(len(full_ds) * val_ratio)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])

    # Wrap with transform
    train_ds.dataset.transform = VideoAugmentation()
    val_ds.dataset.transform = None  # no aug in validation

    return train_ds, val_ds

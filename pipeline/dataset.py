# pipeline/dataset.py

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from pipeline.augmentations import VideoAugmentation


class VideoClipDataset(Dataset):
    def __init__(self, metadata_csv, label_list, clip_len=5, fps=30, transform=None):
        """
        metadata_csv: path to CSV with columns
          clip_path,label,start_frame,end_frame,source_video
        label_list:  list of possible labels (strings)
        clip_len:    number of seconds to load per clip
        fps:         frames per second
        transform:   optional callable on raw frame numpy array
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.label_list = label_list
        self.clip_len = clip_len
        self.fps = fps
        self.transform = transform

        # debug counters
        self.total_clips = 0
        self.black_fallbacks = 0

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # increment total attempted
        self.total_clips += 1

        row = self.metadata.iloc[idx]
        clip_path = row["clip_path"]
        label_vector = self._get_label_vector(row["label"])

        # Always read from start of the trimmed clip
        frames = self._load_clip(clip_path, start_time=0)

        # detect pure-black fallback (no frames read)
        if np.all(frames[0] == 0):
            self.black_fallbacks += 1

        if self.transform:
            frames = self.transform(frames)

        # Convert to tensor: (C, T, H, W), normalize to [0,1]
        clip_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        label_tensor = torch.tensor(label_vector, dtype=torch.float32)

        return clip_tensor, label_tensor

    def _load_clip(self, clip_path, start_time):
        """
        Load clip_len seconds of frames from clip_path starting at start_time.
        Resize to 112×112, pad if too short, or return black frames if none.
        Returns np array of shape (T, H, W, C).
        """
        cap = cv2.VideoCapture(clip_path)
        start_frame = int(start_time * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        total_frames = int(self.clip_len * self.fps)
        frames = []
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()

        # If no frames could be read, return black frames
        if not frames:
            black = np.zeros((112, 112, 3), dtype=np.uint8)
            frames = [black] * total_frames
        # Else if shorter than expected, pad with last frame
        elif len(frames) < total_frames:
            frames.extend([frames[-1]] * (total_frames - len(frames)))

        return np.stack(frames, axis=0)

    def _get_label_vector(self, label_str):
        """
        Convert single-label string into a one-hot vector.
        """
        vector = [0] * len(self.label_list)
        label = label_str.strip()
        if label in self.label_list:
            vector[self.label_list.index(label)] = 1
        return vector


def load_train_val_datasets(csv_path, label_list, val_ratio=0.2):
    """
    Returns (train_ds, val_ds), splitting by val_ratio.
    Applies VideoAugmentation() on training set.
    """
    full_ds = VideoClipDataset(csv_path, label_list)

    val_len = int(len(full_ds) * val_ratio)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])

    train_ds.dataset.transform = VideoAugmentation()
    val_ds.dataset.transform = None

    return train_ds, val_ds

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pipeline.augmentations import VideoAugmentation


class VideoClipDataset(Dataset):
    def __init__(self, metadata_csv, label_list, clip_len=5, fps=30, transform=None):
        """
        metadata_csv: path to CSV with columns
          clip_path,label,start_frame,end_frame,source_video
        label_list:  list of possible labels (strings)
        clip_len:    number of seconds to load per clip
        fps:         frames per second
        transform:   optional callable on raw frame array
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.label_list = label_list
        self.clip_len = clip_len
        self.fps = fps
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1) get the clip file path
        clip_path = row["clip_path"]

        # 2) get start_frame (integer) and convert to seconds
        start_frame = int(row["start_frame"])
        start_time = start_frame / self.fps

        # 3) build one-hot label vector
        label_vector = self._get_label_vector(row["label"])

        # 4) load the frames for this clip
        frames = self._load_clip(clip_path, start_time)

        # 5) apply any video‐level transforms/augmentations
        if self.transform:
            frames = self.transform(frames)

        # 6) convert to tensor: (C, T, H, W), normalize to [0,1]
        clip_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        label_tensor = torch.tensor(label_vector, dtype=torch.float32)

        return clip_tensor, label_tensor

    def _load_clip(self, clip_path, start_time):
        """
        Uses cv2 to load clip_len seconds of frames from clip_path,
        starting at start_time (seconds).
        Resizes each frame to 112×112 and pads if too few frames.
        Returns a numpy array of shape (T, H, W, C).
        """
        cap = cv2.VideoCapture(clip_path)
        start_frame = int(start_time * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        total_frames = int(self.clip_len * self.fps)
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()

        # pad with last frame if clip was shorter than expected
        if frames and len(frames) < total_frames:
            frames.extend([frames[-1]] * (total_frames - len(frames)))

        return np.stack(frames, axis=0)

    def _get_label_vector(self, label_str):
        """
        Converts a single-label string into a one-hot vector.
        label_str: e.g. "W Possession"
        """
        vector = [0] * len(self.label_list)
        label = label_str.strip()
        if label in self.label_list:
            vector[self.label_list.index(label)] = 1
        return vector


def load_train_val_datasets(csv_path, label_list, val_ratio=0.2):
    """
    Returns a (train_ds, val_ds) tuple of Dataset objects,
    splitting the full VideoClipDataset by val_ratio.
    Applies VideoAugmentation() to the training split.
    """
    full_ds = VideoClipDataset(csv_path, label_list)

    # compute split sizes
    val_len = int(len(full_ds) * val_ratio)
    train_len = len(full_ds) - val_len

    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])

    # attach augmentations
    train_ds.dataset.transform = VideoAugmentation()
    val_ds.dataset.transform = None

    return train_ds, val_ds

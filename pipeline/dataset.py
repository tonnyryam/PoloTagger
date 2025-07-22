# pipeline/dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchvision.io import read_video


def load_train_val_datasets(csv_path, label_list, val_ratio=0.2):
    df = pd.read_csv(csv_path)
    num_list = sorted(int(lbl.lstrip("#")) for lbl in label_list if lbl.startswith("#"))
    full_ds = PoloClipDataset(df, num_list)
    n_val = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    return random_split(full_ds, [n_train, n_val])


class PoloClipDataset(Dataset):
    def __init__(self, df, num_list, num_frames=4):
        self.df = df.reset_index(drop=True)
        self.num_list = num_list
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video, _, _ = read_video(row["clip_path"])
        clip = self._sample_clip(video)

        raw_lbl = row["label"].strip()
        pres, team = self._parse_number_label(raw_lbl)

        return clip, (pres, team)

    def _sample_clip(self, video):
        T_total = video.shape[0]
        if T_total >= self.num_frames:
            indices = torch.linspace(0, T_total - 1, self.num_frames).long()
        else:
            indices = torch.arange(self.num_frames) % T_total
        frames = video[indices]
        frames = frames.permute(3, 0, 1, 2).float().div(255.0)
        return frames

    def _parse_number_label(self, raw):
        N = len(self.num_list)
        pres = torch.zeros(N, dtype=torch.float32)
        team = torch.full((N,), -1, dtype=torch.long)

        if raw.startswith("#"):
            n = int(raw.lstrip("#"))
            if n in self.num_list:
                i = self.num_list.index(n)
                pres[i] = 1.0
        elif raw.startswith("W "):
            parts = raw.split()
            if parts[1].isdigit():
                n = int(parts[1])
                if n in self.num_list:
                    i = self.num_list.index(n)
                    pres[i] = 1.0
                    team[i] = 1
        elif raw.startswith("D "):
            parts = raw.split()
            if parts[1].isdigit():
                n = int(parts[1])
                if n in self.num_list:
                    i = self.num_list.index(n)
                    pres[i] = 1.0
                    team[i] = 2

        return pres, team

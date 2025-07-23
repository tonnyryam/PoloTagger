import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchvision.io import read_video
from torchvision import transforms


def load_train_val_datasets(csv_path, label_list, val_ratio=0.2):
    df = pd.read_csv(csv_path)
    num_list = sorted(int(lbl.lstrip("#")) for lbl in label_list if lbl.startswith("#"))
    full_ds = PoloClipDataset(df, num_list)
    n_val = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    return random_split(full_ds, [n_train, n_val])


class PoloClipDataset(Dataset):
    """
    Returns per item:
      clip: Tensor[C, T, H, W] of floats in [0,1]
      (pres_target, team_target)
    """

    def __init__(self, df, num_list, num_frames=2):
        self.df = df.reset_index(drop=True)
        self.num_list = num_list
        self.num_frames = num_frames
        # resize each frame to the R3D-18 default input size:
        self.resize = transforms.Resize((112, 112))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames, _, _ = read_video(row["clip_path"], pts_unit="sec")
        clip = self._sample_clip(frames)

        raw = row["label"].strip()
        pres, team = self._parse_number_label(raw)
        return clip, (pres, team)

    def _sample_clip(self, video):
        T = video.shape[0]
        if T >= self.num_frames:
            idxs = torch.linspace(0, T - 1, self.num_frames).long()
        else:
            idxs = torch.arange(self.num_frames) % T

        # Select frames and apply resize + normalize
        selected = video[idxs]  # [T, H, W, 3]
        resized = [
            self.resize(frame.permute(2, 0, 1)).float() / 255.0 for frame in selected
        ]  # list of [C, 112, 112]

        # Stack into [C, T, H, W]
        clip = torch.stack(resized, dim=1)
        return clip

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

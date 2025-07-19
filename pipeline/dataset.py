import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchvision.io import read_video


def load_train_val_datasets(csv_path, label_list, val_ratio=0.2):
    """
    Reads the CSV, constructs a PoloClipDataset, and splits into train/val.
    Returns: (train_dataset, val_dataset)
    """
    df = pd.read_csv(csv_path)
    # Build the list of jersey‐numbers (integers) from your label_list
    num_list = sorted(int(lbl.lstrip("#")) for lbl in label_list if lbl.startswith("#"))
    full_ds = PoloClipDataset(df, num_list)
    n_val = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    return train_ds, val_ds


class PoloClipDataset(Dataset):
    """
    Each item returns:
      clip: Tensor[C, T, H, W] of floats in [0,1]
      (pres_target, team_target):
        pres_target: FloatTensor[N] of 0/1 presence flags
        team_target: LongTensor[N] of {-1=unknown, 0=absent, 1=white, 2=dark}
    """

    def __init__(self, df, num_list, num_frames=16):
        self.df = df.reset_index(drop=True)
        self.num_list = num_list
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # === load video file (using clip_path column) ===
        video, _, _ = read_video(row["clip_path"])  # [T, H, W, 3], uint8
        clip = self._sample_clip(video)  # → [C, T, H, W], float32 in [0,1]

        # === parse the single raw-label for this clip ===
        raw_lbl = row["label"].strip()  # e.g. "#12" or "W 6/5" or "D Possession"
        pres, team = self._parse_number_label(raw_lbl)

        return clip, (pres, team)

    def _sample_clip(self, video):
        """
        Uniformly samples self.num_frames from the T frames and
        rearranges to [C, T, H, W] float32 tensor.
        """
        T_total = video.shape[0]
        if T_total >= self.num_frames:
            indices = torch.linspace(0, T_total - 1, self.num_frames).long()
        else:
            # pad by looping
            indices = torch.arange(self.num_frames) % T_total
        frames = video[indices]  # [T, H, W, 3]
        frames = frames.permute(3, 0, 1, 2).float() / 255.0
        return frames

    def _parse_number_label(self, raw):
        """
        Builds the two targets for the one raw label:
          - pres_target[n] = 1 if #n is in this clip
          - team_target[n] = 1 if white, 2 if dark, -1 if only number-known, 0 if absent
        """
        N = len(self.num_list)
        pres = torch.zeros(N, dtype=torch.float32)
        team = torch.full((N,), -1, dtype=torch.long)  # -1 = unknown/team-agnostic

        # handle “#12”
        if raw.startswith("#"):
            n = int(raw.lstrip("#"))
            if n in self.num_list:
                i = self.num_list.index(n)
                pres[i] = 1.0
                team[i] = -1

        # handle “W 6/5”
        elif raw.startswith("W "):
            parts = raw.split()
            if parts[1].isdigit():
                n = int(parts[1])
                if n in self.num_list:
                    i = self.num_list.index(n)
                    pres[i] = 1.0
                    team[i] = 1

        # handle “D 6/5”
        elif raw.startswith("D "):
            parts = raw.split()
            if parts[1].isdigit():
                n = int(parts[1])
                if n in self.num_list:
                    i = self.num_list.index(n)
                    pres[i] = 1.0
                    team[i] = 2

        return pres, team

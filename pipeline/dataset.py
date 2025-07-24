import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


def load_train_val_datasets(csv_path, label_list, val_ratio=0.2):
    """
    Reads a CSV whose `clip_path` column now points to preprocessed .pt files
    (each a [T, C, H, W] uint8 tensor), and returns a train/val split.
    """
    df = pd.read_csv(csv_path)
    # Build sorted list of numeric cap labels (e.g. [1,2,...,14])
    num_list = sorted(int(lbl.lstrip("#")) for lbl in label_list if lbl.startswith("#"))
    full_ds = PoloClipDataset(df, num_list)
    n_val = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    return random_split(full_ds, [n_train, n_val])


class PoloClipDataset(Dataset):
    """
    Each item is:
      clip: FloatTensor[C, T, H, W] in [0,1]
      (pres_target, team_target)
    """

    def __init__(self, df: pd.DataFrame, num_list: list[int], num_frames: int = 2):
        self.df = df.reset_index(drop=True)
        self.num_list = num_list
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Load the small .pt tensor saved in preprocess: shape [T, C, H, W], dtype=torch.uint8
        clip = torch.load(row["clip_path"])
        # Convert to float [0,1] and select evenly spaced frames:
        T = clip.shape[0]
        if T >= self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        else:
            indices = torch.arange(self.num_frames) % T
        # Select and reorder to [C, T, H, W]
        sub = clip[indices].float().div(255.0)  # [num_frames, C, H, W]
        clip = sub.permute(1, 0, 2, 3)  # [C, T, H, W]

        # Parse label into presence and team tensors
        raw = row["label"].strip()
        pres = torch.zeros(len(self.num_list), dtype=torch.float32)
        team = torch.full((len(self.num_list),), -1, dtype=torch.long)

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

        return clip, (pres, team)

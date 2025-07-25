import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchvision.io import read_video


def load_train_val_datasets(
    csv_path: str, label_list: list[str], val_ratio: float = 0.2
):
    """
    Reads a CSV with a 'clip_path' column pointing to either .pt or .mp4 files,
    returns a randomized train/val split of PoloClipDataset.
    """
    df = pd.read_csv(csv_path)
    # Build sorted list of numeric cap labels, e.g. [1,2,...]
    num_list = sorted(int(lbl.lstrip("#")) for lbl in label_list if lbl.startswith("#"))
    full_ds = PoloClipDataset(df, num_list)

    n_val = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    return random_split(full_ds, [n_train, n_val])


class PoloClipDataset(Dataset):
    """
    Each item returns:
      clip: FloatTensor[C, T, H, W] in [0,1]
      (presence_target, team_target)
    Supports both preprocessed .pt tensors and raw .mp4 videos.
    """

    def __init__(self, df: pd.DataFrame, num_list: list[int], num_frames: int = 2):
        self.df = df.reset_index(drop=True)
        self.num_list = num_list
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["clip_path"]

        # Load frames: either a pickled tensor or decode video
        if path.lower().endswith(".pt"):
            # [T, C, H, W]
            frames = torch.load(path)
        else:
            # Decode full-resolution video: returns [T, H, W, 3]
            frames, _, _ = read_video(path, pts_unit="sec")
            # Convert to [T, C, H, W]
            frames = frames.permute(0, 3, 1, 2)

        # Sample num_frames evenly
        T = frames.shape[0]
        if T >= self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        else:
            indices = torch.arange(self.num_frames).remainder(T)

        selected = frames[indices]  # [num_frames, C, H, W]
        # Reorder to [C, T, H, W] and normalize
        clip = selected.permute(1, 0, 2, 3).float().div(255.0)

        # Parse label
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

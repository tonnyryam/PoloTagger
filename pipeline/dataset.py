import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import av
import numpy as np


def decode_n_frames(path: str, num_frames: int) -> torch.Tensor:
    """
    Decode exactly `num_frames` evenly-spaced frames from a video file at `path`.
    Returns a [num_frames, C, H, W] uint8 tensor.
    """
    # First pass: count total frames
    container = av.open(path)
    total = 0
    for _ in container.decode(video=0):
        total += 1
    container.close()

    # Compute frame indices to extract
    indices = torch.linspace(0, total - 1, num_frames).long().tolist()
    idx_set = set(indices)

    # Second pass: decode only selected frames
    container = av.open(path)
    out_frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in idx_set:
            arr = np.array(frame.to_image())  # [H, W, 3]
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            out_frames.append(tensor)
            if len(out_frames) == len(indices):
                break
    container.close()

    return torch.stack(out_frames, dim=0)  # [num_frames, 3, H, W]


class PoloClipDataset(Dataset):
    """
    Dataset returning:
      clip: FloatTensor[C, T, H, W] normalized to [0,1]
      (presence_target, team_target)
    Supports raw .mp4 via PyAV and pre-saved .pt tensors.
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

        if path.lower().endswith(".pt"):
            # Load pre-saved tensor [T, C, H, W]
            frames = torch.load(path)
            T = frames.shape[0]
            # Sample evenly if more frames than needed
            if T >= self.num_frames:
                idxs = torch.linspace(0, T - 1, self.num_frames).long()
            else:
                idxs = torch.arange(self.num_frames).remainder(T)
            frames = frames[idxs]
        else:
            # Decode only required frames from video
            frames = decode_n_frames(path, self.num_frames)

        # Reorder [num_frames, C, H, W] -> [C, T, H, W] and normalize
        clip = frames.permute(1, 0, 2, 3).float().div(255.0)

        # Parse label into presence and team targets
        raw = row.get("label", "").strip()
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


def load_train_val_datasets(
    csv_path: str, label_list: list[str], val_ratio: float = 0.2
):
    """
    Read a CSV and return a train/validation split of PoloClipDataset.
    """
    df = pd.read_csv(csv_path)
    num_list = sorted(int(lbl.lstrip("#")) for lbl in label_list if lbl.startswith("#"))
    full_ds = PoloClipDataset(df, num_list)
    n_val = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    return random_split(full_ds, [n_train, n_val])

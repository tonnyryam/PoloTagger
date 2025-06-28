import torchvision.transforms as T
import random
import torch

class VideoAugmentation:
    def __init__(self):
        self.spatial = T.RandomApply([
            T.RandomHorizontalFlip(p=1.0),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ], p=0.7)

        self.normalize = T.Normalize(mean=[0.45], std=[0.225])  # or adapt to 3-channel if needed

    def __call__(self, clip):
        # clip shape: (C, T, H, W) → apply transforms per-frame
        frames = []
        for t in range(clip.shape[1]):
            frame = clip[:, t, :, :]  # shape (C, H, W)
            frame = T.ToPILImage()(frame.cpu())
            frame = self.spatial(frame)
            frame = T.ToTensor()(frame)
            frame = self.normalize(frame)
            frames.append(frame)

        augmented = torch.stack(frames, dim=1)  # (C, T, H, W)
        return augmented

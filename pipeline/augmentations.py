import torchvision.transforms as T
import random
import torch
from PIL import Image

class VideoAugmentation:
    def __init__(self):
        self.frame_aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
            T.RandomRotation(degrees=10)
        ])
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, clip):
        # clip shape: (C, T, H, W)
        frames = []
        for t in range(clip.shape[1]):
            frame = clip[:, t, :, :]
            frame = T.ToPILImage()(frame.cpu())
            frame = self.frame_aug(frame)
            frame = T.ToTensor()(frame)
            frame = self.normalize(frame)
            frames.append(frame)
        return torch.stack(frames, dim=1)

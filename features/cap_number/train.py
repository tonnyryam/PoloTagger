import torch
import numpy as np
import cv2
from features.cap_number.identifier import identify_numbers_in_frame

def add_feature_training(model, clips, labels, label_list):
    penalties = []
    idx = label_list.index("W Possession")

    for i in range(clips.shape[0]):
        # Use the last frame in the clip for number detection
        frame = clips[i, :, -1].permute(1, 2, 0).detach().cpu().numpy()
        frame = (frame * 255).astype(np.uint8)

        caps = identify_numbers_in_frame(frame)
        w_caps = caps.get("W", [])
        d_caps = caps.get("D", [])

        # Example: Penalize if W Possession but no white team cap numbers detected
        if labels[i][idx] == 1 and len(w_caps) == 0:
            penalties.append(0.3)

    if penalties:
        return sum(penalties) / len(penalties)
    return 0.0
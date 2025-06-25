import torch
import numpy as np
import cv2
from features.cap_number.identifier import identify_numbers_in_frame, load_detector

# Load detector once when this module is imported
detector = load_detector("best_yolo_cap_model.pt")  # update with your actual model path if needed

def add_feature_training(model, clips, labels, label_list):
    """
    Apply penalties during training based on cap number presence.

    Args:
        model: The model being trained (not used here).
        clips: A batch of video clips (B x C x T x H x W).
        labels: Corresponding labels for each clip.
        label_list: List of label names in order.

    Returns:
        A scalar penalty to add to loss.
    """
    penalties = []
    idx = label_list.index("W Possession")  # Only penalize missing white caps during W Possession

    for i in range(clips.shape[0]):
        # Use the last frame of the clip (T-1) to identify caps
        frame = clips[i, :, -1].permute(1, 2, 0).detach().cpu().numpy()
        frame = (frame * 255).astype(np.uint8)

        caps = identify_numbers_in_frame(frame, model=detector)
        w_caps = caps.get("W", [])
        d_caps = caps.get("D", [])

        # Penalize if W Possession label is on, but no white caps detected
        if labels[i][idx] == 1 and len(w_caps) == 0:
            penalties.append(0.3)

    if penalties:
        return sum(penalties) / len(penalties)
    return 0.0

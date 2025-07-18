from features.cap_number.identifier import identify_numbers_in_frame
import re

def extract_tagged_cap_numbers(label_str):
    """
    From a string like "#12,#34" extract [12, 34].
    """
    return [int(n) for n in re.findall(r"#(\d+)", label_str)]

def add_feature_training(model, clips, labels, label_list):
    """
    Penalize when the model is supposed to have 'Possession' for a team
    (white or dark) but our detector didn’t see any caps of that team.
    """
    # Build a label → index mapping from the canonical list
    label_to_idx = {lbl: idx for idx, lbl in enumerate(label_list)}
    idx_W_pos = label_to_idx.get("W Possession", None)
    idx_D_pos = label_to_idx.get("D Possession", None)

    penalties = []

    for i in range(clips.size(0)):
        # Reconstruct the last frame of the clip as H×W×3 uint8
        frame = clips[i, :, -1].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype("uint8")

        # Detect caps and their brightness
        caps = identify_numbers_in_frame(frame)

        # Ground-truth tagged numbers from the label vector
        tag_str = ",".join(
            label_list[j] for j, v in enumerate(labels[i]) if v == 1
        )
        tagged = extract_tagged_cap_numbers(tag_str)

        # Brightnesses of tagged numbers
        vs = [d["avg_v"] for d in caps["meta"] if d["number"] in tagged]
        if not vs:
            continue

        # Determine which team (W vs D) by average brightness
        team = "W" if sum(vs) / len(vs) > 128 else "D"

        # If that team's possession label exists and is active but no caps found, penalize
        if team == "W" and idx_W_pos is not None and labels[i][idx_W_pos] == 1 and not caps.get("W", []):
            penalties.append(0.3)
        if team == "D" and idx_D_pos is not None and labels[i][idx_D_pos] == 1 and not caps.get("D", []):
            penalties.append(0.3)

    return sum(penalties) / len(penalties) if penalties else 0.0

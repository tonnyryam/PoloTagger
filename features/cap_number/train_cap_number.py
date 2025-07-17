from features.cap_number.identifier import identify_numbers_in_frame

def extract_tagged_cap_numbers(label_str):
    import re
    # e.g. "#12,#34" → [12, 34]
    return [int(n) for n in re.findall(r"#(\d+)", label_str)]

def add_feature_training(model, clips, labels, label_list):
    """
    A dummy example: penalize if the expected cap numbers (embedded
    in your label_list entries via '#<num>') aren’t detected.
    """
    idx = label_list.index("W Possession")  # adjust this to the label that carries tags
    penalties = []

    for i in range(clips.size(0)):
        # Convert the last frame of the clip back to H×W×3 uint8
        frame = clips[i, :, -1].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype("uint8")

        caps   = identify_numbers_in_frame(frame)
        tag_str = ",".join([label_list[j] for j,v in enumerate(labels[i]) if v == 1])
        tagged  = extract_tagged_cap_numbers(tag_str)

        # Look up brightness for any tagged number
        vs = [d["avg_v"] for d in caps["meta"] if d["number"] in tagged]
        if vs:
            team = "W" if sum(vs) / len(vs) > 128 else "D"
            # if model predicts W Possession but no 'W' caps found, penalize
            if team and labels[i][idx] == 1 and not caps[team]:
                penalties.append(0.3)

    return sum(penalties) / len(penalties) if penalties else 0.0

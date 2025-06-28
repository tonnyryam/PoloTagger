from features.cap_number.identifier import identify_numbers_in_frame

def extract_tagged_cap_numbers(label_str):
    # Example: "#5, #7" -> [5, 7]
    import re
    return [int(n) for n in re.findall(r"#(\d+)", label_str)]

def add_feature_training(model, clips, labels, label_list):
    idx = label_list.index("W Possession")  # Adjust label as needed
    penalties = []

    for i in range(clips.shape[0]):
        frame = clips[i, :, -1].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype("uint8")

        caps = identify_numbers_in_frame(frame)
        tag_str = ",".join([label_list[j] for j, v in enumerate(labels[i]) if v == 1])
        tagged_caps = extract_tagged_cap_numbers(tag_str)

        brightness_vals = [d["avg_v"] for d in caps["meta"] if d["number"] in tagged_caps]
        if brightness_vals:
            avg_brightness = sum(brightness_vals) / len(brightness_vals)
            inferred_team = "W" if avg_brightness > 128 else "D"
        else:
            inferred_team = None

        if inferred_team and labels[i][idx] == 1 and len(caps[inferred_team]) == 0:
            penalties.append(0.3)

    return sum(penalties) / len(penalties) if penalties else 0.0

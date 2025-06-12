import os
import pandas as pd
import xml.etree.ElementTree as ET

def parse_sportscode_xml(xml_path):
    """
    Parse Sportscode XML file and return a list of (start_time, end_time, label)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    events = []

    for instance in root.iter('Instance'):
        try:
            start_time = float(instance.find('Start').text)
            end_time = float(instance.find('End').text)
            label = instance.find('Label').text.strip()
            events.append((start_time, end_time, label))
        except:
            continue
    return events

def build_clip_index(video_path, xml_path, output_dir, clip_len=5):
    """
    Build a CSV index of clips and their corresponding labels
    """
    os.makedirs(output_dir, exist_ok=True)
    events = parse_sportscode_xml(xml_path)
    rows = []

    for i, (start, end, label) in enumerate(events):
        # Define a clip file name
        clip_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_clip_{i:04d}.mp4"
        clip_path = os.path.join(output_dir, clip_name)

        # Use ffmpeg to extract the clip
        cmd = f"ffmpeg -ss {start:.2f} -i '{video_path}' -t {clip_len} -c:v libx264 -an '{clip_path}' -y"
        os.system(cmd)

        rows.append({"clip_path": clip_path, "labels": label})

    return pd.DataFrame(rows)

def main():
    video_path = "data/raw/game1.mp4"
    xml_path = "data/raw/game1.xml"
    output_dir = "data/clips"
    output_csv = "data/metadata/clip_index.csv"

    df = build_clip_index(video_path, xml_path, output_dir, clip_len=5)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved clip index to {output_csv}")

if __name__ == "__main__":
    main()
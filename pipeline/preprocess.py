import os
import xml.etree.ElementTree as ET
import csv
import subprocess
from pathlib import Path

# Configuration
CLIP_LEN = 5.0  # seconds
STRIDE = 2.5  # seconds

# Directory paths
INPUT_DIR = Path("data/raw_videos")
OUTPUT_DIR = Path("data/clips")
LABELS_DIR = Path("data/labels")
METADATA_DIR = Path("data/metadata")
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def parse_sportscode_xml(xml_path):
    """Parse Sportscode XML to extract event start, end, and label."""
    events = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for instance in root.findall(".//instance"):
        label = instance.find("label").text.strip()
        start_time = float(instance.find("start").text)
        end_time = float(instance.find("end").text)
        events.append({"start": start_time, "end": end_time, "label": label})
    
    return events

def match_events_to_clip(clip_start, clip_end, events):
    """Return list of event labels that overlap with a given clip."""
    tags = []
    for event in events:
        if (event["start"] < clip_end) and (event["end"] > clip_start):
            tags.append(event["label"])
    return list(set(tags))  # remove duplicates

def generate_clips_and_labels():
    """Generate labeled video clips from full-game videos and Sportscode XML."""
    metadata_file = METADATA_DIR / "clip_index.csv"
    with open(metadata_file, mode='w', newline='') as csvfile:
        fieldnames = ['clip_path', 'source_video', 'clip_start', 'clip_end', 'labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for video_file in INPUT_DIR.glob("*.mp4"):
            video_name = video_file.stem
            output_subdir = OUTPUT_DIR / video_name
            output_subdir.mkdir(parents=True, exist_ok=True)

            xml_file = LABELS_DIR / f"{video_name}.xml"
            if not xml_file.exists():
                print(f"⚠️  No XML tag file found for {video_name}, skipping.")
                continue

            events = parse_sportscode_xml(xml_file)

            # Get video duration
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            duration = float(result.stdout)

            clip_index = 0
            clip_start = 0.0
            while clip_start + CLIP_LEN <= duration:
                clip_end = clip_start + CLIP_LEN
                labels = match_events_to_clip(clip_start, clip_end, events)

                output_filename = f"clip_{clip_index:04d}.mp4"
                output_path = output_subdir / output_filename

                subprocess.run([
                    "ffmpeg", "-loglevel", "error", "-y",
                    "-ss", str(clip_start),
                    "-t", str(CLIP_LEN),
                    "-i", str(video_file),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    str(output_path)
                ])

                writer.writerow({
                    "clip_path": str(output_path),
                    "source_video": video_name,
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                    "labels": ";".join(labels)
                })

                clip_index += 1
                clip_start += STRIDE

if __name__ == "__main__":
    generate_clips_and_labels()

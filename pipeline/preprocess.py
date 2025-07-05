import os
import argparse
import pandas as pd
import xml.etree.ElementTree as ET
from moviepy.editor import VideoFileClip
from docx import Document

def parse_xml(xml_path, fps):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    clips = []
    for instance in root.findall(".//instance"):
        label = instance.find("label").text
        start = int(instance.find("start_frame").text)
        end = int(instance.find("end_frame").text)
        clips.append((label, start, end))
    return clips

def parse_docx(docx_path, fps):
    doc = Document(docx_path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    clips = []
    i = 0
    while i < len(lines) - 1:
        try:
            time_parts = lines[i].split()
            if len(time_parts) == 2:
                start = int(float(time_parts[0]) * fps)
                end = int(float(time_parts[1]) * fps)
                label = lines[i + 1]
                clips.append((label, start, end))
                i += 2
            else:
                i += 1
        except ValueError:
            i += 1
    return clips

def extract_clip(video_path, out_path, start_frame, end_frame, fps):
    try:
        print(f"[DEBUG] Extracting: {out_path}")
        start_time = start_frame / fps
        end_time = end_frame / fps
        print(f"[DEBUG] Time range: {start_time:.2f}s to {end_time:.2f}s")
        clip = VideoFileClip(video_path)
        print(f"[DEBUG] Video duration: {clip.duration:.2f}s")
        if end_time > clip.duration:
            print(f"[WARN] Trimming end_time {end_time:.2f}s to {clip.duration:.2f}s")
            end_time = clip.duration
        if start_time >= end_time:
            print(f"[ERROR] Invalid clip range: start {start_time:.2f}s >= end {end_time:.2f}s")
            return
        subclip = clip.subclip(start_time, end_time)
        subclip.write_videofile(out_path, codec="libx264", audio=False, verbose=True)
    except Exception as e:
        print(f"[ERROR] ffmpeg/moviepy failed to write {out_path}: {e}")

def preprocess_all(input_dir, out_dir, metadata_csv, clip_len=5, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)

    entries = []
    existing_videos = set()

    if os.path.exists(metadata_csv):
        try:
            df_existing = pd.read_csv(metadata_csv)
            if not df_existing.empty:
                existing_videos = set(df_existing['source_video'].unique())
                entries = df_existing.to_dict("records")
            else:
                print(f"[WARN] Existing metadata file is empty: {metadata_csv}")
        except pd.errors.EmptyDataError:
            print(f"[WARN] Could not parse metadata CSV (empty or malformed): {metadata_csv}")

    for file in os.listdir(input_dir):
        if file.endswith(".mp4"):
            base = os.path.splitext(file)[0]
            if base in existing_videos:
                print(f"⏩ Skipping already processed video: {base}")
                continue

            video_path = os.path.join(input_dir, f"{base}.mp4")
            xml_path = os.path.join(input_dir, f"{base}.xml")
            docx_path = os.path.join(input_dir, f"{base}.docx")

            if os.path.exists(xml_path):
                clips = parse_xml(xml_path, fps)
            elif os.path.exists(docx_path):
                clips = parse_docx(docx_path, fps)
            else:
                print(f"⚠️ Skipping {base}: No annotation file (.xml or .docx) found.")
                continue

            print(f"📦 Processing: {base}")
            video_duration = VideoFileClip(video_path).duration
            for label, start, end in clips:
                start_time = start / fps
                end_time = end / fps
                if start_time >= video_duration:
                    print(f"[WARN] Skipping clip starting at {start_time:.2f}s — beyond video end.")
                    continue
                safe_label = label.replace(" ", "_").replace("/", "_")
                clip_filename = f"{safe_label}_{base}_{start}.mp4"
                out_path = os.path.join(out_dir, clip_filename)
                extract_clip(video_path, out_path, start, end, fps)
                entries.append({
                    "clip_path": out_path,
                    "label": label,
                    "start_frame": start,
                    "end_frame": end,
                    "source_video": base
                })

    pd.DataFrame(entries).to_csv(metadata_csv, index=False)
    print(f"✅ Updated metadata saved to {metadata_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw", help="Folder containing .mp4 and annotation files")
    parser.add_argument("--out_dir", default="data/clips", help="Where to store extracted clips")
    parser.add_argument("--metadata_csv", default="data/metadata/clip_index.csv", help="Output metadata file")
    parser.add_argument("--clip_len", type=int, default=5, help="Clip length (used if splitting instead of tags)")
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    args = parser.parse_args()

    preprocess_all(args.input_dir, args.out_dir, args.metadata_csv, args.clip_len, args.fps)
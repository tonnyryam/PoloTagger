import os
import re
import argparse
import traceback
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from docx import Document
from moviepy.editor import VideoFileClip

def extract_clip(video_path, out_path, start_frame, end_frame, fps):
    start = start_frame / fps
    duration = (end_frame - start_frame) / fps
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-an",
        out_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print(f"[ERROR] ffmpeg failed for {out_path}: {e}")

def parse_docx(docx_path, fps):
    print(f"[DEBUG] Parsing DOCX with 4-line label blocks: {docx_path}")
    try:
        doc = Document(docx_path)
        raw_text = "\\n".join([p.text.strip() for p in doc.paragraphs])
        chunks = re.split(r'\\n{2,}', raw_text)
        print(f"[DEBUG] Found {len(chunks)} blocks")

        clips = []
        for i, chunk in enumerate(chunks):
            lines = [line.strip() for line in chunk.splitlines() if line.strip()]
            if len(lines) < 4:
                continue
            try:
                index = lines[0]  # ignored
                start = float(lines[1])
                end = float(lines[2])
                label = lines[3]
                start_frame = int(start * fps)
                end_frame = int(end * fps)
                clips.append((label, start_frame, end_frame))
                print(f"[DEBUG] Parsed block {i}: {label} [{start:.2f} → {end:.2f}]")
            except Exception as e:
                print(f"[WARN] Could not parse block {i}: {e}")
        print(f"[INFO] ✅ Parsed {len(clips)} clips from DOCX")
        return clips
    except Exception as e:
        print(f"[ERROR] Failed to parse DOCX file: {e}")
        traceback.print_exc()
        return []

def preprocess_all(input_dir, out_dir, metadata_csv, clip_len, fps):
    entries = []
    seen = set()

    if os.path.exists(metadata_csv):
        try:
            df_existing = pd.read_csv(metadata_csv)
            seen = set(df_existing['clip_path'].tolist())
        except Exception as e:
            print(f"[WARN] Could not parse metadata CSV (empty or malformed): {metadata_csv}")

    for fname in os.listdir(input_dir):
        if not fname.endswith(".mp4"):
            continue
        base = fname[:-4]
        video_path = os.path.join(input_dir, fname)
        docx_path = os.path.join(input_dir, base + ".docx")
        if not os.path.exists(docx_path):
            print(f"[WARN] No .docx file for {base}")
            continue

        print(f"📦 Processing: {base}")
        try:
            clips = parse_docx(docx_path, fps)
            clip = VideoFileClip(video_path)
            video_duration = clip.duration
            clip.close()
        except Exception as e:
            print(f"[ERROR] Failed to load video or parse .docx for {base}: {e}")
            continue

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for label, start, end in clips:
                start_time = start / fps
                end_time = end / fps
                if start_time >= video_duration:
                    continue
                safe_label = label.replace(" ", "_").replace("/", "_").replace("#", "")
                clip_filename = f"{safe_label}_{base}_{start}.mp4"
                out_path = os.path.join(out_dir, clip_filename)
                if out_path in seen:
                    continue
                entries.append({
                    "clip_path": out_path,
                    "label": label,
                    "start_frame": start,
                    "end_frame": end,
                    "source_video": base
                })
                futures.append(executor.submit(extract_clip, video_path, out_path, start, end, fps))

            for f in as_completed(futures):
                f.result()

    if entries:
        df = pd.DataFrame(entries)
        os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)
        df.to_csv(metadata_csv, index=False)
        print(f"✅ Updated metadata saved to {metadata_csv}")
    else:
        print("[WARN] No new clips created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--clip_len", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    preprocess_all(args.input_dir, args.out_dir, args.metadata_csv, args.clip_len, args.fps)
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
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-an",
        out_path,
    ]
    try:
        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
    except Exception as e:
        print(f"[ERROR] ffmpeg failed for {out_path}: {e}")


def parse_docx(docx_path, fps):
    print(f"[DEBUG] Parsing DOCX with 4-line label blocks: {docx_path}")
    try:
        doc = Document(docx_path)

        # join with real newlines
        raw_text = "\n".join([p.text.strip() for p in doc.paragraphs])
        chunks = re.split(r"\n{2,}", raw_text)
        print(f"[DEBUG] Found {len(chunks)} blocks after splitting")

        clips = []
        for i, chunk in enumerate(chunks):
            lines = [line.strip() for line in chunk.splitlines() if line.strip()]
            if len(lines) < 4:
                print(f"[DEBUG] Block {i} skipped (need ≥4 lines, got {len(lines)})")
                continue

            start, end = float(lines[1]), float(lines[2])
            label = lines[3]
            if start >= end:
                print(f"[DEBUG] Block {i} skipped (start ≥ end)")
                continue

            start_frame = int(start * fps)
            end_frame = int(end * fps)
            clips.append((label, start_frame, end_frame))
            print(f"[DEBUG] Block {i} → '{label}' frames [{start_frame}→{end_frame}]")
        print(f"[INFO] ✅ Parsed {len(clips)} clips from DOCX")
        return clips

    except Exception as e:
        print(f"[ERROR] Failed to parse DOCX file: {e}")
        traceback.print_exc()
        return []


def preprocess_all(input_dir, out_dir, metadata_csv, clip_len, fps):
    entries = []
    seen = set()
    existing_df = None

    # load existing CSV
    if os.path.exists(metadata_csv):
        try:
            existing_df = pd.read_csv(metadata_csv)
            seen = set(existing_df["clip_path"].tolist())
            print(f"[INFO] Loaded existing CSV with {len(existing_df)} entries")
        except Exception:
            existing_df = None

    for fname in os.listdir(input_dir):
        if not fname.endswith(".mp4"):
            continue
        base = fname[:-4]
        video_path = os.path.join(input_dir, fname)
        docx_path = os.path.join(input_dir, base + ".docx")
        if not os.path.exists(docx_path):
            print(f"[WARN] No .docx for {base}")
            continue

        print(f"📦 Processing: {base}")
        try:
            clips = parse_docx(docx_path, fps)
            clip = VideoFileClip(video_path)
            video_duration = clip.duration
            clip.close()
            print(f"[DEBUG] Video duration: {video_duration}s")
        except Exception as e:
            print(f"[ERROR] Failed loading {base}: {e}")
            continue

        max_end_frame = int(video_duration * fps)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for label, start_frame, end_frame in clips:
                start_time = start_frame / fps
                if start_time >= video_duration:
                    print(f"[DEBUG] Skipping '{label}' (start ≥ duration)")
                    continue

                # clamp end_frame to video length
                if end_frame > max_end_frame:
                    print(
                        f"[DEBUG] Clamping '{label}' end_frame {end_frame}→{max_end_frame}"
                    )
                    end_frame = max_end_frame

                if end_frame <= start_frame:
                    print(
                        f"[DEBUG] Skipping '{label}' after clamping (no positive duration)"
                    )
                    continue

                safe_label = label.replace(" ", "_").replace("/", "_").replace("#", "")
                clip_filename = f"{safe_label}_{base}_{start_frame}.mp4"
                out_path = os.path.join(out_dir, clip_filename)

                if out_path in seen:
                    print(f"[DEBUG] Skipping duplicate: {out_path}")
                    continue

                print(f"[DEBUG] Queueing clip: {out_path}")
                entries.append(
                    {
                        "clip_path": out_path,
                        "label": label,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "source_video": base,
                    }
                )
                futures.append(
                    executor.submit(
                        extract_clip, video_path, out_path, start_frame, end_frame, fps
                    )
                )

            for f in as_completed(futures):
                f.result()

    # write out CSV
    os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)
    if entries:
        new_df = pd.DataFrame(entries)
        final_df = (
            pd.concat([existing_df, new_df], ignore_index=True)
            if existing_df is not None
            else new_df
        )
        final_df.to_csv(metadata_csv, index=False)
        print(f"[INFO] ✅ Metadata saved to {metadata_csv} ({len(entries)} new clips)")
    else:
        if existing_df is None:
            pd.DataFrame(
                columns=[
                    "clip_path",
                    "label",
                    "start_frame",
                    "end_frame",
                    "source_video",
                ]
            ).to_csv(metadata_csv, index=False)
            print(f"[INFO] ✅ Created empty CSV at {metadata_csv}")
        else:
            print(f"[INFO] ✅ No new clips; CSV unchanged ({len(existing_df)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--clip_len", type=float, default=5)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    preprocess_all(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        metadata_csv=args.metadata_csv,
        clip_len=args.clip_len,
        fps=args.fps,
    )

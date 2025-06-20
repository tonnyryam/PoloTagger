import argparse
import os
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
from moviepy.editor import VideoFileClip

def extract_clips_from_video(video_path, xml_path, output_dir, clip_len=5, fps=30):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    metadata = []

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Loaded video: {basename}, {total_frames} frames @ {actual_fps:.2f} FPS")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for instance in root.findall(".//instance"):
        label = instance.find("label").text.strip()
        start_frame = int(instance.find("start_frame").text)
        end_frame = int(instance.find("end_frame").text)

        # Clip center logic
        center_frame = (start_frame + end_frame) // 2
        half_clip = (clip_len * fps) // 2
        clip_start = max(center_frame - half_clip, 0)
        clip_end = min(center_frame + half_clip, total_frames - 1)

        video.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        frames = []
        for _ in range(clip_end - clip_start):
            success, frame = video.read()
            if not success:
                break
            resized = cv2.resize(frame, (224, 224))  # Standard size
            frames.append(resized)

        if len(frames) < 1:
            continue

        clip_filename = f"{label}_{basename}_{clip_start}.mp4"
        out_path = os.path.join(output_dir, clip_filename)

        height, width, _ = frames[0].shape
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for f in frames:
            writer.write(f)
        writer.release()

        metadata.append({
            "clip_path": out_path,
            "label": label,
            "start_frame": clip_start,
            "end_frame": clip_end,
            "source_video": video_path
        })

    video.release()
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input .mp4 video file")
    parser.add_argument("--xml", required=True, help="Path to Sportscode XML tag file")
    parser.add_argument("--out_dir", default="data/clips", help="Where to store extracted clips")
    parser.add_argument("--metadata_csv", default="data/metadata/clip_index.csv", help="Where to save the metadata CSV")
    parser.add_argument("--clip_len", type=int, default=5, help="Clip length in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for extraction")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.metadata_csv), exist_ok=True)
    metadata = extract_clips_from_video(
        video_path=args.video,
        xml_path=args.xml,
        output_dir=args.out_dir,
        clip_len=args.clip_len,
        fps=args.fps
    )

    df = pd.DataFrame(metadata)
    df.to_csv(args.metadata_csv, index=False)
    print(f"Saved metadata to {args.metadata_csv}")

if __name__ == "__main__":
    main()

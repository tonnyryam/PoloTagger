import os
import argparse
import pandas as pd

# Ensure PyTorch is available for saving .pt clips
try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required to save .pt clips")

from torchvision.io import read_video


def preprocess_all(input_dir, out_dir, metadata_csv, num_frames, fps):
    """
    Read raw .mp4 videos & segmentation metadata, sample `num_frames` evenly spaced full-resolution
    frames from each clip, and save them as .pt tensors. Generates a new CSV suffixed with `_pt.csv`.

    Args:
      input_dir: directory of raw videos
      out_dir: directory to write .pt clips
      metadata_csv: path to original clip_index.csv
      num_frames: number of frames to sample per clip
      fps: frames per second to interpret start/end
    """
    # Load existing metadata
    df = pd.read_csv(metadata_csv)
    out_records = []

    os.makedirs(out_dir, exist_ok=True)

    for idx, row in df.iterrows():
        path = row["clip_path"]
        base = os.path.splitext(os.path.basename(path))[0]

        # Decode full clip
        frames, _, _ = read_video(path, pts_unit="sec")  # [T, H, W, 3]
        T = frames.shape[0]

        # select evenly spaced indices
        indices = torch.linspace(0, T - 1, num_frames).long()

        # pick & reorder to [num_frames, C, H, W]
        sel = frames[indices].permute(0, 3, 1, 2).contiguous()

        # save to .pt
        out_name = f"{base}.pt"
        out_path = os.path.join(out_dir, out_name)
        torch.save(sel, out_path)

        # record new path for the CSV
        rec = row.to_dict()
        rec["clip_path"] = out_path
        out_records.append(rec)

    # write new CSV pointing at .pt clips
    df_pt = pd.DataFrame(out_records)
    csv_pt = metadata_csv.replace(".csv", "_pt.csv")
    df_pt.to_csv(csv_pt, index=False)
    print(f"[INFO] Saved {len(df_pt)} pt clips and metadata → {csv_pt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-sample frames and save as .pt tensors"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory of raw .mp4 videos"
    )
    parser.add_argument("--out_dir", required=True, help="Directory to write .pt clips")
    parser.add_argument(
        "--metadata_csv", required=True, help="Path to original clip_index.csv"
    )
    parser.add_argument(
        "--num_frames", type=int, default=2, help="Number of frames to sample per clip"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="FPS to interpret video timestamps"
    )
    args = parser.parse_args()

    preprocess_all(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        metadata_csv=args.metadata_csv,
        num_frames=args.num_frames,
        fps=args.fps,
    )

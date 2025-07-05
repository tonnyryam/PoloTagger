import os
import argparse
import pandas as pd
import xml.etree.ElementTree as ET
from moviepy.editor import VideoFileClip
from docx import Document
import traceback

def parse_xml(xml_path, fps):
    """Parse XML file with extensive debugging"""
    print(f"[DEBUG] Parsing XML: {xml_path}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        print(f"[DEBUG] XML root tag: {root.tag}")
        print(f"[DEBUG] XML root attributes: {root.attrib}")
        
        # Print the structure to understand the XML format
        print(f"[DEBUG] First 10 elements in XML:")
        for i, elem in enumerate(root.iter()):
            if i > 10:
                break
            print(f"  {i}: {elem.tag} - {elem.text} - {elem.attrib}")
        
        clips = []
        instances = root.findall(".//instance")
        print(f"[DEBUG] Found {len(instances)} instances with xpath './/instance'")
        
        if len(instances) == 0:
            # Try alternative xpath patterns
            alternative_patterns = [
                ".//Instance", 
                ".//item", 
                ".//clip", 
                ".//event",
                ".//*[contains(local-name(), 'instance')]",
                ".//*[contains(local-name(), 'Instance')]"
            ]
            
            for pattern in alternative_patterns:
                alt_instances = root.findall(pattern)
                print(f"[DEBUG] Pattern '{pattern}' found {len(alt_instances)} elements")
                if len(alt_instances) > 0:
                    instances = alt_instances
                    break
        
        for i, instance in enumerate(instances):
            print(f"[DEBUG] Processing instance {i+1}/{len(instances)}")
            
            # Try different ways to find label
            label_elem = instance.find("label")
            if label_elem is None:
                label_elem = instance.find("Label")
            if label_elem is None:
                label_elem = instance.find("name")
            if label_elem is None:
                label_elem = instance.find("Name")
            
            # Try different ways to find start_frame
            start_elem = instance.find("start_frame")
            if start_elem is None:
                start_elem = instance.find("start")
            if start_elem is None:
                start_elem = instance.find("Start")
            if start_elem is None:
                start_elem = instance.find("startFrame")
            
            # Try different ways to find end_frame
            end_elem = instance.find("end_frame")
            if end_elem is None:
                end_elem = instance.find("end")
            if end_elem is None:
                end_elem = instance.find("End")
            if end_elem is None:
                end_elem = instance.find("endFrame")
            
            if label_elem is None:
                print(f"[ERROR] No label found in instance {i+1}")
                print(f"[DEBUG] Instance structure: {ET.tostring(instance, encoding='unicode')}")
                continue
                
            if start_elem is None:
                print(f"[ERROR] No start_frame found in instance {i+1}")
                print(f"[DEBUG] Instance structure: {ET.tostring(instance, encoding='unicode')}")
                continue
                
            if end_elem is None:
                print(f"[ERROR] No end_frame found in instance {i+1}")
                print(f"[DEBUG] Instance structure: {ET.tostring(instance, encoding='unicode')}")
                continue
            
            try:
                label = label_elem.text.strip() if label_elem.text else ""
                start = int(float(start_elem.text))
                end = int(float(end_elem.text))
                
                print(f"[DEBUG] Found clip: '{label}' from frame {start} to {end}")
                clips.append((label, start, end))
                
            except (ValueError, AttributeError) as e:
                print(f"[ERROR] Could not parse values from instance {i+1}: {e}")
                print(f"[DEBUG] Label: {label_elem.text if label_elem is not None else 'None'}")
                print(f"[DEBUG] Start: {start_elem.text if start_elem is not None else 'None'}")
                print(f"[DEBUG] End: {end_elem.text if end_elem is not None else 'None'}")
                continue
        
        print(f"[DEBUG] Successfully parsed {len(clips)} clips from XML")
        return clips
        
    except ET.ParseError as e:
        print(f"[ERROR] XML parsing error: {e}")
        # Try to read first few lines to see the format
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:10]
                print(f"[DEBUG] First 10 lines of XML file:")
                for i, line in enumerate(lines):
                    print(f"  {i+1}: {line.strip()}")
        except Exception as read_error:
            print(f"[ERROR] Could not read XML file: {read_error}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error parsing XML: {e}")
        traceback.print_exc()
        return []

def parse_docx(docx_path, fps):
    """Parse DOCX file with extensive debugging"""
    print(f"[DEBUG] Parsing DOCX: {docx_path}")
    
    try:
        doc = Document(docx_path)
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        print(f"[DEBUG] Found {len(lines)} non-empty lines in DOCX")
        
        # Print first 20 lines for debugging
        print(f"[DEBUG] First 20 lines:")
        for i, line in enumerate(lines[:20]):
            print(f"  {i+1}: '{line}'")
        
        clips = []
        i = 0
        while i < len(lines) - 1:
            try:
                current_line = lines[i]
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                
                print(f"[DEBUG] Processing line {i+1}: '{current_line}'")
                
                # Try to parse time information
                time_parts = current_line.split()
                
                if len(time_parts) >= 2:
                    try:
                        # Try different time formats
                        start_time = float(time_parts[0])
                        end_time = float(time_parts[1])
                        
                        start_frame = int(start_time * fps)
                        end_frame = int(end_time * fps)
                        
                        label = next_line
                        
                        print(f"[DEBUG] Found clip: '{label}' from {start_time}s to {end_time}s (frames {start_frame}-{end_frame})")
                        clips.append((label, start_frame, end_frame))
                        i += 2
                        
                    except ValueError as e:
                        print(f"[DEBUG] Could not parse time from '{current_line}': {e}")
                        i += 1
                else:
                    print(f"[DEBUG] Line doesn't have 2 time parts: '{current_line}'")
                    i += 1
                    
            except Exception as e:
                print(f"[ERROR] Error processing line {i+1}: {e}")
                i += 1
        
        print(f"[DEBUG] Successfully parsed {len(clips)} clips from DOCX")
        return clips
        
    except Exception as e:
        print(f"[ERROR] Error parsing DOCX file: {e}")
        traceback.print_exc()
        return []

def extract_clip(video_path, out_path, start_frame, end_frame, fps):
    """Extract clip with better error handling"""
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
            clip.close()
            return False
            
        if start_time < 0:
            print(f"[WARN] Negative start time {start_time:.2f}s, setting to 0")
            start_time = 0
        
        subclip = clip.subclip(start_time, end_time)
        subclip.write_videofile(out_path, codec="libx264", audio=False, verbose=False, logger=None)
        
        clip.close()
        subclip.close()
        
        print(f"[SUCCESS] Extracted clip: {out_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to extract clip {out_path}: {e}")
        traceback.print_exc()
        return False

def preprocess_all(input_dir, out_dir, metadata_csv, clip_len=5, fps=30):
    """Main preprocessing function with extensive debugging"""
    print(f"[DEBUG] Starting preprocessing...")
    print(f"[DEBUG] Input directory: {input_dir}")
    print(f"[DEBUG] Output directory: {out_dir}")
    print(f"[DEBUG] Metadata CSV: {metadata_csv}")
    print(f"[DEBUG] FPS: {fps}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return
    
    # List all files in input directory
    all_files = os.listdir(input_dir)
    print(f"[DEBUG] Files in input directory: {all_files}")
    
    # Filter for video files
    video_files = [f for f in all_files if f.endswith('.mp4')]
    print(f"[DEBUG] Video files found: {video_files}")
    
    # Check for annotation files
    xml_files = [f for f in all_files if f.endswith('.xml')]
    docx_files = [f for f in all_files if f.endswith('.docx')]
    print(f"[DEBUG] XML files found: {xml_files}")
    print(f"[DEBUG] DOCX files found: {docx_files}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)

    entries = []
    existing_videos = set()

    # Load existing metadata if available
    if os.path.exists(metadata_csv):
        try:
            df_existing = pd.read_csv(metadata_csv)
            if not df_existing.empty:
                existing_videos = set(df_existing['source_video'].unique())
                entries = df_existing.to_dict("records")
                print(f"[DEBUG] Loaded {len(entries)} existing entries")
            else:
                print(f"[WARN] Existing metadata file is empty: {metadata_csv}")
        except pd.errors.EmptyDataError:
            print(f"[WARN] Could not parse metadata CSV (empty or malformed): {metadata_csv}")

    # Process each video file
    for video_file in video_files:
        base = os.path.splitext(video_file)[0]
        print(f"\n[DEBUG] Processing video: {base}")
        
        if base in existing_videos:
            print(f"⏩ Skipping already processed video: {base}")
            continue

        video_path = os.path.join(input_dir, f"{base}.mp4")
        xml_path = os.path.join(input_dir, f"{base}.xml")
        docx_path = os.path.join(input_dir, f"{base}.docx")
        
        print(f"[DEBUG] Looking for annotations:")
        print(f"  XML path: {xml_path} - exists: {os.path.exists(xml_path)}")
        print(f"  DOCX path: {docx_path} - exists: {os.path.exists(docx_path)}")

        clips = []
        
        if os.path.exists(xml_path):
            print(f"[DEBUG] Processing XML file: {xml_path}")
            clips = parse_xml(xml_path, fps)
        elif os.path.exists(docx_path):
            print(f"[DEBUG] Processing DOCX file: {docx_path}")
            clips = parse_docx(docx_path, fps)
        else:
            print(f"⚠️ Skipping {base}: No annotation file (.xml or .docx) found.")
            continue

        if not clips:
            print(f"[WARN] No clips extracted from annotation file for {base}")
            continue

        print(f"📦 Processing {len(clips)} clips for: {base}")
        
        try:
            video_clip = VideoFileClip(video_path)
            video_duration = video_clip.duration
            video_clip.close()
            print(f"[DEBUG] Video duration: {video_duration:.2f}s")
        except Exception as e:
            print(f"[ERROR] Could not load video {video_path}: {e}")
            continue

        successful_clips = 0
        for i, (label, start, end) in enumerate(clips):
            print(f"\n[DEBUG] Processing clip {i+1}/{len(clips)}: '{label}'")
            
            start_time = start / fps
            end_time = end / fps
            
            if start_time >= video_duration:
                print(f"[WARN] Skipping clip starting at {start_time:.2f}s — beyond video end ({video_duration:.2f}s).")
                continue
            
            # Create safe filename
            safe_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
            safe_label = "".join(c for c in safe_label if c.isalnum() or c in "._-")
            
            clip_filename = f"{safe_label}_{base}_{start}.mp4"
            out_path = os.path.join(out_dir, clip_filename)
            
            if extract_clip(video_path, out_path, start, end, fps):
                entries.append({
                    "clip_path": out_path,
                    "label": label,
                    "start_frame": start,
                    "end_frame": end,
                    "source_video": base,
                    "video_path": video_path,
                    "start_time": start_time,
                    "labels": label  # Add this for compatibility with your training script
                })
                successful_clips += 1
            else:
                print(f"[ERROR] Failed to extract clip {i+1}")
        
        print(f"[SUCCESS] Successfully extracted {successful_clips}/{len(clips)} clips for {base}")

    # Save metadata
    if entries:
        df = pd.DataFrame(entries)
        df.to_csv(metadata_csv, index=False)
        print(f"✅ Updated metadata saved to {metadata_csv} with {len(entries)} entries")
    else:
        print(f"[WARN] No clips were extracted, metadata not updated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw", help="Folder containing .mp4 and annotation files")
    parser.add_argument("--out_dir", default="data/clips", help="Where to store extracted clips")
    parser.add_argument("--metadata_csv", default="data/metadata/clip_index.csv", help="Output metadata file")
    parser.add_argument("--clip_len", type=int, default=5, help="Clip length (used if splitting instead of tags)")
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    args = parser.parse_args()

    preprocess_all(args.input_dir, args.out_dir, args.metadata_csv, args.clip_len, args.fps)
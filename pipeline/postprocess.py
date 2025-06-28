import argparse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from collections import defaultdict


def parse_predictions(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = defaultdict(list)

    for instance in root.findall("instance"):
        label = instance.find("label").text
        start = int(instance.find("start_frame").text)
        end = int(instance.find("end_frame").text)
        confidence = float(instance.find("confidence").text) if instance.find("confidence") is not None else 1.0
        source_video = instance.find("source_video").text if instance.find("source_video") is not None else "Unknown"
        annotations[source_video].append((label, start, end, confidence))

    return annotations


def merge_and_filter(annotations, min_duration, min_conf, merge_gap):
    filtered = defaultdict(list)

    for video, items in annotations.items():
        items.sort(key=lambda x: (x[0], x[1]))  # sort by label, then start
        merged = []

        for label in set(x[0] for x in items):
            group = [x for x in items if x[0] == label and x[3] >= min_conf]
            group.sort(key=lambda x: x[1])
            temp = []

            for item in group:
                if not temp:
                    temp.append(item)
                    continue
                prev = temp[-1]
                if item[1] - prev[2] <= merge_gap:
                    # merge
                    new_item = (label, prev[1], max(prev[2], item[2]), max(prev[3], item[3]))
                    temp[-1] = new_item
                else:
                    temp.append(item)

            for label, start, end, conf in temp:
                if end - start >= min_duration:
                    filtered[video].append((label, start, end, conf))

    return filtered


def export_to_sportscode(annotations, output_path, fps):
    root = Element("annotations")
    for video, items in annotations.items():
        for label, start, end, conf in items:
            inst = SubElement(root, "instance")
            SubElement(inst, "start_time").text = f"{start / fps:.2f}"
            SubElement(inst, "end_time").text = f"{end / fps:.2f}"
            SubElement(inst, "label").text = label
            SubElement(inst, "source_video").text = video

    ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"✅ Exported Sportscode-compatible XML to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, help="Path to input predictions XML")
    parser.add_argument("--out", required=True, help="Output Sportscode XML path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--min_duration", type=int, default=15, help="Minimum duration (in frames) for a label to be kept")
    parser.add_argument("--min_conf", type=float, default=0.5, help="Minimum confidence threshold")
    parser.add_argument("--merge_gap", type=int, default=30, help="Max gap (in frames) to merge same-label clips")
    args = parser.parse_args()

    annotations = parse_predictions(args.xml)
    filtered = merge_and_filter(annotations, args.min_duration, args.min_conf, args.merge_gap)
    export_to_sportscode(filtered, args.out, args.fps)


if __name__ == "__main__":
    main()

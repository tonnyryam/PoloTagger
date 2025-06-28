import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from xml.etree.ElementTree import Element, SubElement, ElementTree
from pipeline.dataset import VideoClipDataset
from torchvision.models.video import r3d_18
from features.cap_number.identifier import load_detector, identify_numbers_in_frame

label_list = [
    "W Possession", "W Turn Over", "D Possession", "D CA", "D Turn Over",
    "Referee", "W CA", "W DEXC", "W 6/5", "D Goals", "D Shots", "D 5/6",
    "W FCO", "W AG", "W New 20", "D Center", "W Shots", "W Center",
    "D DEXC", "D 6/5", "W 5/6", "D Penalty", "W Penalty", "W Goals",
    "D AG", "D FCO", "W Time Out"
]

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def evaluate_model(model, dataloader, device, threshold=0.5, show_caps=False):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    clip_infos = []

    with torch.no_grad():
        for i, (clips, labels) in enumerate(dataloader):
            clips = clips.to(device)
            outputs = model(clips)
            probs = sigmoid(outputs).cpu()
            preds = (probs > threshold).int()

            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

            for j in range(len(clips)):
                row = dataloader.dataset.metadata.iloc[i * dataloader.batch_size + j]
                clip_infos.append({
                    "video": row["source_video"],
                    "start": int(row["start_frame"]),
                    "end": int(row["end_frame"]),
                    "clip_path": row["clip_path"]
                })

                if show_caps:
                    frame = clips[j, :, -1].permute(1, 2, 0).cpu().numpy()
                    frame = (frame * 255).astype("uint8")
                    caps = identify_numbers_in_frame(frame)
                    print(f"Clip {j} Caps Detected: {caps}")

    return all_preds, all_probs, all_labels, clip_infos

def export_to_xml(preds, probs, clip_infos, label_list, output_path):
    annotations = Element("annotations")
    for pred, prob, info in zip(preds, probs, clip_infos):
        for label_idx, value in enumerate(pred):
            if value == 1:
                instance = SubElement(annotations, "instance")
                SubElement(instance, "start_frame").text = str(info["start"])
                SubElement(instance, "end_frame").text = str(info["end"])
                SubElement(instance, "label").text = label_list[label_idx]
                SubElement(instance, "source_video").text = info["video"]
                SubElement(instance, "confidence").text = f"{float(prob[label_idx]):.4f}"
    tree = ElementTree(annotations)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to clip_index.csv")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for multilabel outputs")
    parser.add_argument("--export_xml", help="Optional path to save HUDL-compatible XML file")
    parser.add_argument("--yolo_model", default="best_yolo_cap_model.pt", help="Path to YOLOv8 model for cap detection")
    parser.add_argument("--digit_model", default="digit_classifier.pth", help="Path to digit classifier model")
    parser.add_argument("--show_caps", action="store_true", help="Print cap numbers for each clip during inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_detector(args.yolo_model, digit_model_path=args.digit_model)

    dataset = VideoClipDataset(args.csv, label_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = r3d_18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(label_list))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    print("Running inference...")
    preds, probs, labels, clip_infos = evaluate_model(model, dataloader, device, args.threshold, show_caps=args.show_caps)

    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, target_names=label_list, zero_division=0))

    if args.export_xml:
        print(f"Exporting predictions to XML: {args.export_xml}")
        export_to_xml(preds, probs, clip_infos, label_list, args.export_xml)

if __name__ == "__main__":
    main()

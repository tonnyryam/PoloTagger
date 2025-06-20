import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from pipeline.dataset import VideoClipDataset
from torchvision.models.video import r3d_18
from evaluate_predictions import export_to_xml

label_list = [
    "W Possession", "W Turn Over", "D Possession", "D CA", "D Turn Over",
    "Referee", "W CA", "W DEXC", "W 6/5", "D Goals", "D Shots", "D 5/6",
    "W FCO", "W AG", "W New 20", "D Center", "W Shots", "W Center",
    "D DEXC", "D 6/5", "W 5/6", "D Penalty", "W Penalty", "W Goals",
    "D AG", "D FCO", "W Time Out"
]

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for clips, labels, paths in dataloader:
            clips = clips.to(device)
            outputs = model(clips)
            probs = sigmoid(outputs).cpu()
            preds = (probs > threshold).int()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_paths.extend(paths)

    return all_preds, all_labels, all_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to clip_index.csv")
    parser.add_argument("--clips", required=True, help="Path to folder with video clips")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for multilabel outputs")
    parser.add_argument("--export_xml", help="Optional path to save HUDL-compatible XML file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VideoClipDataset(args.csv, label_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = r3d_18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(label_list))
    model.load_state_dict(torch.load(args.model))
    model.to(device)

    print("Running inference...")
    preds, labels, paths = evaluate_model(model, dataloader, device, args.threshold)

    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, target_names=label_list, zero_division=0))

    if args.export_xml:
        print(f"Exporting predictions to XML: {args.export_xml}")
        export_to_xml(preds, paths, label_list, args.export_xml)

if __name__ == "__main__":
    main()

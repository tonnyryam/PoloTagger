import argparse
import os
import importlib.util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from pipeline.dataset import load_train_val_datasets
from features.cap_number.identifier import load_detector

# List of 27 action/label classes
label_list = [
    "W Possession",
    "W Turn Over",
    "D Possession",
    "D CA",
    "D Turn Over",
    "Referee",
    "W CA",
    "W DEXC",
    "W 6/5",
    "D Goals",
    "D Shots",
    "D 5/6",
    "W FCO",
    "W AG",
    "W New 20",
    "D Center",
    "W Shots",
    "W Center",
    "D DEXC",
    "D 6/5",
    "W 5/6",
    "D Penalty",
    "W Penalty",
    "W Goals",
    "D AG",
    "D FCO",
    "W Time Out",
]


def load_feature_trainers(features_folder):
    """Discover feature‐training hooks under features_folder."""
    trainers = []
    for root, _, files in os.walk(features_folder):
        if "train.py" in files:
            train_path = os.path.join(root, "train.py")
            spec = importlib.util.spec_from_file_location("train", train_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "add_feature_training"):
                trainers.append(module.add_feature_training)
    return trainers


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    feature_trainers,
    label_list,
    num_epochs=5,
):
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        total_train_loss = 0.0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            # apply any feature‐specific penalties
            penalty = sum(
                trainer(model, clips, labels, label_list)
                for trainer in feature_trainers
            )
            total = loss + penalty
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            total_train_loss += total.item() * clips.size(0)
        avg_train = total_train_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch}/{num_epochs}] Train Loss: {avg_train:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(clips)
                vloss = criterion(outputs, labels)
                total_val_loss += vloss.item() * clips.size(0)
        avg_val = total_val_loss / len(val_loader.dataset)
        print(f"[Epoch {epoch}/{num_epochs}]   Val Loss: {avg_val:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/metadata/clip_index.csv",
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--features",
        default="features",
        help="Directory containing feature‐training modules",
    )
    parser.add_argument(
        "--out",
        default="models/r3d_18_final.pth",
        help="Path to save the trained R3D model",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--yolo_model",
        default="yolov5s.pt",
        help="YOLO specifier (auto-downloaded via torch.hub)",
    )
    parser.add_argument(
        "--digit_model",
        default="mnist-onnx",
        help="Digit model specifier (ONNX/HF hub)",
    )
    args = parser.parse_args()

    print(f"[INFO] YOLO model specifier:  {args.yolo_model}")
    print(f"[INFO] Digit model specifier: {args.digit_model}")

    # — verify inputs —
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"Metadata CSV not found: {args.csv}")
    if not os.path.isdir(args.features):
        raise FileNotFoundError(f"Features directory not found: {args.features}")

    # ensure output folder exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # load YOLO & digit detectors (handles hub downloading)
    load_detector(args.yolo_model, digit_model_path=args.digit_model)

    # prepare datasets & loaders
    train_ds, val_ds = load_train_val_datasets(args.csv, label_list, val_ratio=0.2)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # build R3D-18 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(label_list))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    feature_trainers = load_feature_trainers(args.features)

    # run the training loop
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        feature_trainers,
        label_list,
        num_epochs=args.epochs,
    )

    # save final weights
    torch.save(model.state_dict(), args.out)
    print(f"[INFO] Model saved to {args.out}")


if __name__ == "__main__":
    main()

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
import os
import importlib.util
from pipeline.dataset import load_train_val_datasets

label_list = [
    "W Possession", "W Turn Over", "D Possession", "D CA", "D Turn Over",
    "Referee", "W CA", "W DEXC", "W 6/5", "D Goals", "D Shots", "D 5/6",
    "W FCO", "W AG", "W New 20", "D Center", "W Shots", "W Center",
    "D DEXC", "D 6/5", "W 5/6", "D Penalty", "W Penalty", "W Goals",
    "D AG", "D FCO", "W Time Out"
]

def load_feature_trainers(features_folder):
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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, feature_trainers, label_list, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)

            outputs = model(clips)
            loss = criterion(outputs, labels)

            # Feature-specific penalties
            feature_penalty = sum(trainer(model, clips, labels, label_list) for trainer in feature_trainers)
            total_loss = loss + feature_penalty

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(clips)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}] Val Loss:   {avg_val_loss:.4f}")
        model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/metadata/clip_index.csv", help="Path to label CSV file")
    parser.add_argument("--features", default="features", help="Folder containing feature modules")
    parser.add_argument("--out", default="models/r3d_18_final.pth", help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train_ds, val_ds = load_train_val_datasets(args.csv, label_list, val_ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(label_list))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    feature_trainers = load_feature_trainers(args.features)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, feature_trainers, label_list, num_epochs=args.epochs)
    torch.save(model.state_dict(), args.out)
    print(f"Model saved to {args.out}")

if __name__ == "__main__":
    main()

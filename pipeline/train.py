import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from pipeline.dataset import VideoClipDataset

import importlib.util
import os

label_list = [
    "W Possession", "W Turn Over", "D Possession", "D CA", "D Turn Over",
    "Referee", "W CA", "W DEXC", "W 6/5", "D Goals", "D Shots", "D 5/6",
    "W FCO", "W AG", "W New 20", "D Center", "W Shots", "W Center",
    "D DEXC", "D 6/5", "W 5/6", "D Penalty", "W Penalty", "W Goals",
    "D AG", "D FCO", "W Time Out"
]

def load_feature_trainers(features_dir="features"):
    trainers = []
    for feature_name in os.listdir(features_dir):
        path = os.path.join(features_dir, feature_name, "train.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("train_" + feature_name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "add_feature_training"):
                trainers.append(mod.add_feature_training)
    return trainers

def train_model(model, dataloader, criterion, optimizer, device, feature_trainers, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for clips, labels in dataloader:
            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for trainer in feature_trainers:
                trainer(model, clips, labels)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VideoClipDataset("data/metadata/clip_index.csv", label_list)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(label_list))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    feature_trainers = load_feature_trainers("features")

    train_model(model, dataloader, criterion, optimizer, device, feature_trainers, num_epochs=5)

    torch.save(model.state_dict(), "models/r3d_18.pth")

if __name__ == "__main__":
    main()
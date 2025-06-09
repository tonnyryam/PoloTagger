import torch
import VideoClipDataset
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
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

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


# Replace with your actual label list
label_list = [
    "W Possession", "W Turn Over", "D Possession", "D CA", "D Turn Over",
    "Referee", "W CA", "W DEXC", "W 6/5", "D Goals", "D Shots", "D 5/6",
    "W FCO", "W AG", "W New 20", "D Center", "W Shots", "W Center",
    "D DEXC", "D 6/5", "W 5/6", "D Penalty", "W Penalty", "W Goals",
    "D AG", "D FCO", "W Time Out"
]

# Paths and hyperparameters
metadata_csv_path = "data/metadata/clip_index.csv"
batch_size = 4
num_epochs = 5
learning_rate = 1e-4

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = VideoClipDataset(metadata_csv=metadata_csv_path, label_list=label_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = r3d_18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(label_list))
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
train_model(model, dataloader, criterion, optimizer, device, num_epochs)
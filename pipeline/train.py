import argparse
import os
import sys
import time
import logging
import socket
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18

from data.metadata.labels import label_list  # your generated labels.py
from pipeline.dataset import load_train_val_datasets

# --- Setup logging as before ---
logger = logging.getLogger("PoloTagger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)


# --- Multi‐task model wrapper ---
class PoloMultiTask(nn.Module):
    def __init__(self, backbone: nn.Module, num_numbers: int):
        super().__init__()
        # Remove the original classifier
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # Presence head: binary for each number
        self.pres_head = nn.Linear(feat_dim, num_numbers)
        # Team head: 3‐way for each number
        self.team_head = nn.Linear(feat_dim, num_numbers * 3)

    def forward(self, x):
        feat = self.backbone(x)  # [B, feat_dim]
        pres_logits = self.pres_head(feat)  # [B, N]
        team_logits = self.team_head(feat)  # [B, N*3]
        B = pres_logits.shape[0]
        team_logits = team_logits.view(B, -1, 3)  # [B, N, 3]
        return pres_logits, team_logits


def train_model(model, loader, optimizer, alpha, device):
    model.train()
    total_loss = 0.0
    for clips, (pres_t, team_t) in loader:
        clips = clips.to(device)
        pres_t = pres_t.to(device)
        team_t = team_t.to(device)

        pres_logits, team_logits = model(clips)

        # 1) Presence loss
        loss_pres = F.binary_cross_entropy_with_logits(pres_logits, pres_t)

        # 2) Team loss only where team_t >= 0
        mask = team_t >= 0  # shape [B, N]
        if mask.any():
            valid_logits = team_logits[mask]  # [M, 3]
            valid_labels = team_t[mask]  # [M]
            loss_team = F.cross_entropy(valid_logits, valid_labels)
        else:
            loss_team = torch.tensor(0.0, device=device)

        loss = loss_pres + alpha * loss_team

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clips.size(0)

    return total_loss / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/metadata/clip_index.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="weight for the team‐classification loss",
    )
    args = p.parse_args()

    # SLURM / CPU setup omitted for brevity…

    # Build datasets
    num_list = sorted(int(lbl.lstrip("#")) for lbl in label_list if lbl.startswith("#"))
    train_ds, val_ds = load_train_val_datasets(args.csv, label_list)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, multiprocessing.cpu_count() - 1),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, multiprocessing.cpu_count() - 1),
    )

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = r3d_18(pretrained=True)
    model = PoloMultiTask(base, num_numbers=len(num_list)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_model(model, train_loader, optimizer, args.alpha, device)
        logger.info(
            f"[Epoch {epoch}] Train Loss: {train_loss:.4f} (took {time.time() - t0:.1f}s)"
        )

        # You can add a similar eval_model() for validation here…

    # Save weights (optional format)
    torch.save(model.state_dict(), "models/polo_multitask.pth")
    logger.info("Training complete, model saved.")


if __name__ == "__main__":
    main()

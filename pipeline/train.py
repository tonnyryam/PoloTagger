#!/usr/bin/env python3
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

from data.metadata.labels import label_list
from pipeline.dataset import load_train_val_datasets

# ─── Logger setup ─────────────────────────────────────────────
logger = logging.getLogger("PoloTagger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)


# ─── Multi‐task model definition ──────────────────────────────
class PoloMultiTask(nn.Module):
    def __init__(self, backbone: nn.Module, num_numbers: int):
        super().__init__()
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.pres_head = nn.Linear(feat_dim, num_numbers)
        self.team_head = nn.Linear(feat_dim, num_numbers * 3)

    def forward(self, x):
        feat = self.backbone(x)  # [B, feat_dim]
        pres_logits = self.pres_head(feat)  # [B, N]
        team_logits = self.team_head(feat)  # [B, N*3]
        B = pres_logits.shape[0]
        team_logits = team_logits.view(B, -1, 3)  # [B, N, 3]
        return pres_logits, team_logits


# ─── Training and evaluation loops ───────────────────────────
def train_model(model, loader, optimizer, alpha, device):
    model.train()
    total_loss = 0.0
    for clips, (pres_t, team_t) in loader:
        clips = clips.to(device)
        pres_t = pres_t.to(device)
        team_t = team_t.to(device)

        pres_logits, team_logits = model(clips)

        # Presence loss
        loss_pres = F.binary_cross_entropy_with_logits(pres_logits, pres_t)
        # Team loss on valid entries
        mask = team_t >= 0
        if mask.any():
            valid_logits = team_logits[mask]
            valid_labels = team_t[mask]
            loss_team = F.cross_entropy(valid_logits, valid_labels)
        else:
            loss_team = torch.tensor(0.0, device=device)

        loss = loss_pres + alpha * loss_team

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clips.size(0)

    return total_loss / len(loader.dataset)


def evaluate_model(model, loader, alpha, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for clips, (pres_t, team_t) in loader:
            clips = clips.to(device)
            pres_t = pres_t.to(device)
            team_t = team_t.to(device)

            pres_logits, team_logits = model(clips)

            loss_pres = F.binary_cross_entropy_with_logits(pres_logits, pres_t)
            mask = team_t >= 0
            if mask.any():
                valid_logits = team_logits[mask]
                valid_labels = team_t[mask]
                loss_team = F.cross_entropy(valid_logits, valid_labels)
            else:
                loss_team = torch.tensor(0.0, device=device)

            loss = loss_pres + alpha * loss_team
            total_loss += loss.item() * clips.size(0)

    return total_loss / len(loader.dataset)


# ─── Main entrypoint ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train PoloTagger multi‐task model.")
    parser.add_argument(
        "--csv", default="data/metadata/clip_index.csv", help="Path to metadata CSV"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for the team‐classification loss",
    )
    parser.add_argument("--features", help="(unused) for backwards compatibility")
    parser.add_argument(
        "--out",
        default="models/polo_multitask.pth",
        help="Path to save the trained model",
    )
    args = parser.parse_args()

    # Log environment
    logger.info(f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'N/A')}")
    logger.info(f"HOSTNAME={socket.gethostname()}")

    # Prepare data
    num_list = sorted(int(lbl.lstrip("#")) for lbl in label_list if lbl.startswith("#"))
    train_ds, val_ds = load_train_val_datasets(args.csv, label_list)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        n_workers = max(1, int(slurm_cpus) - 1)
    else:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {n_workers} DataLoader workers")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = r3d_18(pretrained=True)
    model = PoloMultiTask(base, num_numbers=len(num_list)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training + Validation
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_model(model, train_loader, optimizer, args.alpha, device)
        val_loss = evaluate_model(model, val_loader, args.alpha, device)
        elapsed = time.time() - t0

        logger.info(
            f"[Epoch {epoch}/{args.epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (took {elapsed:.1f}s)"
        )

    # Save final model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    logger.info(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()

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
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torchvision.models.video import r3d_18

from data.metadata.labels import label_list
from pipeline.dataset import load_train_val_datasets

# Logger
logger = logging.getLogger("PoloTagger")
logger.setLevel(logging.DEBUG)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
if not logger.handlers:
    logger.addHandler(h)


class PoloMultiTask(nn.Module):
    def __init__(self, backbone, num_numbers):
        super().__init__()
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.pres_head = nn.Linear(feat_dim, num_numbers)
        self.team_head = nn.Linear(feat_dim, num_numbers * 3)

    def forward(self, x):
        feat = self.backbone(x)
        pres_logits = self.pres_head(feat)
        team_logits = self.team_head(feat)
        B = pres_logits.size(0)
        team_logits = team_logits.view(B, -1, 3)
        return pres_logits, team_logits


def train_model(model, loader, optimizer, alpha, device):
    model.train()
    scaler = GradScaler(device_type="cuda")
    total = 0.0

    for clips, (pres_t, team_t) in loader:
        clips = clips.to(device, non_blocking=True)
        pres_t = pres_t.to(device, non_blocking=True)
        team_t = team_t.to(device, non_blocking=True)

        optimizer.zero_grad()
        # Mixed-precision forward/backward
        with autocast():
            p_logit, t_logit = model(clips)
            loss_pres = F.binary_cross_entropy_with_logits(p_logit, pres_t)

            mask = team_t >= 0
            if mask.any():
                v_logit = t_logit[mask]
                v_lab = team_t[mask]
                loss_team = F.cross_entropy(v_logit, v_lab)
            else:
                loss_team = torch.tensor(0.0, device=device)

            loss = loss_pres + alpha * loss_team

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item() * clips.size(0)

    return total / len(loader.dataset)


def evaluate_model(model, loader, alpha, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for clips, (pres_t, team_t) in loader:
            clips = clips.to(device, non_blocking=True)
            pres_t = pres_t.to(device, non_blocking=True)
            team_t = team_t.to(device, non_blocking=True)

            p_logit, t_logit = model(clips)
            loss_pres = F.binary_cross_entropy_with_logits(p_logit, pres_t)

            mask = team_t >= 0
            if mask.any():
                v_logit = t_logit[mask]
                v_lab = team_t[mask]
                loss_team = F.cross_entropy(v_logit, v_lab)
            else:
                loss_team = torch.tensor(0.0, device=device)

            total += (loss_pres + alpha * loss_team).item() * clips.size(0)

    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train PoloTagger multi-task model.")
    parser.add_argument("--csv", default="data/metadata/clip_index.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--features", help="(legacy)")
    parser.add_argument("--out", default="models/polo_multitask.pth")
    args = parser.parse_args()

    logger.info(f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'N/A')}")
    logger.info(f"HOSTNAME={socket.gethostname()}")

    # Determine number of DataLoader workers
    n_workers = 4
    logger.info(f"Using {n_workers} DataLoader workers")

    train_ds, val_ds = load_train_val_datasets(args.csv, label_list)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = r3d_18(weights=r3d_18.default_weights)
    model = PoloMultiTask(
        base, num_numbers=len([lbl for lbl in label_list if lbl.startswith("#")])
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_model(model, train_loader, optimizer, args.alpha, device)
        val_loss = evaluate_model(model, val_loader, args.alpha, device)
        logger.info(
            f"[Epoch {epoch}/{args.epochs}] "
            f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  (took {time.time() - t0:.1f}s)"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    logger.info(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()

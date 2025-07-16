import argparse
import os
import sys
import time
import logging
import multiprocessing
import importlib.util

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18

from pipeline.dataset import load_train_val_datasets
from features.cap_number.identifier import load_detector

# --- Determine repository root and load labels dynamically ---
# Option 2: compute repo_root relative to this file, not cwd
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
labels_path = os.path.join(repo_root, "data", "metadata", "labels.py")
spec = importlib.util.spec_from_file_location("labels", labels_path)
labels_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels_module)
label_list = labels_module.label_list

# --- Setup logging (always to repo’s scripts/train.log) ---
log_dir = os.path.join(repo_root, "scripts")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train.log")

logger = logging.getLogger("PoloTagger")
logger.setLevel(logging.DEBUG)

# Only file handler—all logs go into train.log
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)


def benchmark_loader(ds, bs, n_workers):
    """Load 10 batches and log the average seconds per batch, then exit."""
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    logger.debug(f"[BENCH] batch_size={bs}, num_workers={n_workers}")
    t0 = time.time()
    for i, (clips, labels) in enumerate(loader):
        if i >= 9:
            break
    avg = (time.time() - t0) / 10.0
    logger.debug(f"[BENCH] ≈{avg:.3f}s per batch over 10 batches")
    sys.exit(0)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    feature_trainers,
    label_list,
    num_epochs,
):
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{num_epochs}")
        model.train()
        total_train = 0.0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            penalty = sum(
                trainer(model, clips, labels, label_list)
                for trainer in feature_trainers
            )
            total_loss = loss + penalty
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_train += total_loss.item() * clips.size(0)
        avg_train = total_train / len(train_loader.dataset)
        logger.info(f"[Epoch {epoch}] Train Loss: {avg_train:.4f}")

        # Validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                vout = model(clips)
                vloss = criterion(vout, labels)
                total_val += vloss.item() * clips.size(0)
        avg_val = total_val / len(val_loader.dataset)
        logger.info(f"[Epoch {epoch}]   Val Loss: {avg_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train PoloTagger model.")
    parser.add_argument(
        "--csv", default="data/metadata/clip_index.csv", help="Path to metadata CSV"
    )
    parser.add_argument(
        "--features", default="features", help="Directory containing features"
    )
    parser.add_argument(
        "--out",
        default="models/r3d_18_final.pth",
        help="Output path for the trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--benchmark-data",
        action="store_true",
        help="Run DataLoader timing (10 batches) and exit",
    )
    args = parser.parse_args()

    # Detector loading: feature modules handle their own downloads
    logger.info("Initializing cap_number feature detectors...")
    try:
        load_detector()
        logger.info("cap_number feature ready")
    except Exception as e:
        logger.error(f"Feature initialization failed: {e}")
        sys.exit(1)

    # Validate CSV and features
    if not os.path.isfile(args.csv):
        logger.error(f"Metadata CSV not found: {args.csv}")
        sys.exit(1)
    if not os.path.isdir(args.features):
        logger.error(f"Features dir not found: {args.features}")
        sys.exit(1)

    # Prepare datasets & loaders
    train_ds, val_ds = load_train_val_datasets(args.csv, label_list, val_ratio=0.2)
    n_cpus = multiprocessing.cpu_count()
    n_workers = max(1, n_cpus - 1)
    logger.info(f"Detected {n_cpus} CPU cores → using {n_workers} workers")
    if args.benchmark_data:
        benchmark_loader(train_ds, args.batch_size, n_workers)
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

    # Build and train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(label_list))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    feature_trainers = [load_detector]

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

    # Save final model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    logger.info(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()

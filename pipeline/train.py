import argparse
import os
import sys
import time
import logging
import multiprocessing
import importlib.util
import socket

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18

from pipeline.dataset import load_train_val_datasets
import features.cap_number.identifier as cap_identifier  # was: from features.cap_number.identifier import load_detector, classifier
from features.cap_number.train_cap_number import add_feature_training

# --- Determine repository root and load labels dynamically ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
labels_path = os.path.join(repo_root, "data", "metadata", "labels.py")
spec = importlib.util.spec_from_file_location("labels", labels_path)
labels_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels_mod)
label_list = labels_mod.label_list

# --- Setup logging to stdout only ---
logger = logging.getLogger("PoloTagger")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(sh)

# --- Log initial debug info ---
logger.info(f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'N/A')}")
logger.info(f"HOSTNAME={socket.gethostname()}")
logger.debug(
    f"Python executable: {sys.executable}, version: {sys.version.replace(chr(10), ' ')}"
)
logger.debug(
    f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
)


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
            total = loss + penalty
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            total_train += total.item() * clips.size(0)
        avg_train = total_train / len(train_loader.dataset)
        logger.info(f"[Epoch {epoch}] Train Loss: {avg_train:.4f}")

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
        "--benchmark-data", action="store_true", help="Run DataLoader timing and exit"
    )
    args = parser.parse_args()

    # Initialize detectors once
    logger.info("Initializing cap_number feature detectors.")
    det = cap_identifier.load_detector()
    if det is None:
        logger.error("cap_number YOLO failed to initialize—aborting.")
        sys.exit(1)
    if cap_identifier.classifier is None:
        logger.error("cap_number digit classifier failed to initialize—aborting.")
        sys.exit(1)
    logger.info("cap_number feature ready")

    # Validate inputs
    if not os.path.isfile(args.csv):
        logger.error(f"Metadata CSV not found: {args.csv}")
        sys.exit(1)
    if not os.path.isdir(args.features):
        logger.error(f"Features dir not found: {args.features}")
        sys.exit(1)

    # CPU/worker setup
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            n_cpus = int(slurm_cpus)
            logger.info(f"SLURM_CPUS_PER_TASK={n_cpus}")
        except ValueError:
            n_cpus = multiprocessing.cpu_count()
            logger.warning(f"Invalid SLURM_CPUS_PER_TASK, using {n_cpus}")
    else:
        n_cpus = multiprocessing.cpu_count()
        logger.info(f"Detected CPU count: {n_cpus}")
    n_workers = max(1, n_cpus - 1)
    logger.info(f"Using {n_workers} DataLoader workers")

    # Data loaders
    train_ds, val_ds = load_train_val_datasets(args.csv, label_list, val_ratio=0.2)
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

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(label_list))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    feature_trainers = [add_feature_training]

    # Train!
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

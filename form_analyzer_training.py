"""
form_analyzer_train.py
----------------------
Dataset → DataLoader → Training loop → Evaluation for FormAnalyzerCNN.

Expects:
  - A directory of .pt tensors (shape: T x 51) produced by form_analyzer.py
  - A CSV of labels produced by kinematic_form_analyzer.py with columns:
      filename, overstriding, forward_lean, vertical_bounce

Usage:
  python form_analyzer_train.py --tensor_dir path/to/tensors --labels path/to/training_labels.csv
"""

import argparse
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from form_analyzer import FormAnalyzerCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

LABEL_COLS = ["overstriding", "forward_lean", "vertical_bounce"]
NUM_FEATURES = 51   # 17 keypoints × 3 (x, y, conf)
NUM_CLASSES  = len(LABEL_COLS)
MAX_FRAMES   = 300  # pad / truncate all sequences to this length


# ── Dataset ──────────────────────────────────────────────────────────────────

class RunningFormDataset(Dataset):
    """
    Loads pre-computed .pt keypoint tensors and their kinematic labels.

    Each tensor has shape (T, 51) where T varies by video length.
    We pad or truncate to MAX_FRAMES so batches are uniform.
    Labels are multi-label binary vectors [overstriding, forward_lean, vertical_bounce].
    """

    def __init__(self, tensor_dir: str, labels_csv: str, max_frames: int = MAX_FRAMES):
        self.tensor_dir = Path(tensor_dir)
        self.max_frames = max_frames

        df = pd.read_csv(labels_csv)

        # Keep only rows whose tensor file actually exists on disk
        df["tensor_path"] = df["filename"].apply(
            lambda f: self.tensor_dir / f
        )
        missing = ~df["tensor_path"].apply(lambda p: p.exists())
        if missing.any():
            log.warning(f"Skipping {missing.sum()} rows — tensor files not found.")
        self.df = df[~missing].reset_index(drop=True)

        log.info(f"Dataset ready: {len(self.df)} samples from {tensor_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load tensor (T, 51)
        tensor: torch.Tensor = torch.load(row["tensor_path"], weights_only=True)

        # Pad or truncate to MAX_FRAMES
        T = tensor.shape[0]
        if T >= self.max_frames:
            tensor = tensor[: self.max_frames]
        else:
            pad = torch.zeros(self.max_frames - T, NUM_FEATURES)
            tensor = torch.cat([tensor, pad], dim=0)  # (MAX_FRAMES, 51)

        labels = torch.tensor(
            row[LABEL_COLS].values.astype(np.float32), dtype=torch.float32
        )
        return tensor, labels


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Stack variable-length tensors — padding already applied in Dataset."""
    tensors, labels = zip(*batch)
    return torch.stack(tensors), torch.stack(labels)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for tensors, labels in loader:
        tensors, labels = tensors.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(tensors)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for tensors, labels in loader:
        tensors, labels = tensors.to(device), labels.to(device)
        logits = model(tensors)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    report = classification_report(
        all_labels, all_preds,
        target_names=LABEL_COLS,
        zero_division=0
    )
    return avg_loss, report


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── Device ──
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    # ── Data ──
    dataset = RunningFormDataset(
        tensor_dir=args.tensor_dir,
        labels_csv=args.labels,
        max_frames=args.max_frames,
    )

    val_size   = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ──
    model = FormAnalyzerCNN(num_features=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # BCEWithLogitsLoss handles class imbalance better than BCE + sigmoid
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing — smoothly decays LR without cliff drops
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ──
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, report = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        log.info(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / "best_form_analyzer.pt"
            torch.save(model.state_dict(), save_path)
            log.info(f"  ✓ New best model saved → {save_path}")

        # Print per-class metrics every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            log.info(f"\n{report}")

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Weights saved to: {output_dir / 'best_form_analyzer.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FormAnalyzerCNN")
    parser.add_argument("--tensor_dir", required=True,
                        help="Directory of .pt tensors (T×51)")
    parser.add_argument("--labels",     required=True,
                        help="CSV from kinematic_form_analyzer.py")
    parser.add_argument("--output_dir", default="runs/form_analyzer",
                        help="Where to save weights")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--max_frames", type=int,   default=MAX_FRAMES,
                        help="Pad/truncate all sequences to this many frames")
    args = parser.parse_args()
    main(args)
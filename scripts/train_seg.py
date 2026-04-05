"""
MotivAid Blood Loss — Segmentation Model Training
===================================================
Trains a U-Net (MobileNetV2 encoder) to segment blood stains in images.
This model identifies WHICH PIXELS are blood-stained.

Run on Google Colab with GPU for faster training:
  !python scripts/train_seg.py

Expected output: models/seg_best.pt
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import BloodLossDataset

# ─── Config ───────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
EPOCHS = 60
LR = 1e-4
BEST_MODEL_PATH = os.path.join('models', 'seg_best.pt')

TRAIN_IMAGE_DIR = os.path.join('dataset', 'synthetic_train')
VAL_IMAGE_DIR = os.path.join('dataset', 'synthetic_val')
MASK_DIR = os.path.join('dataset', 'masks')
LABELS_FILE = os.path.join('dataset', 'synthetic_labels.json')

# Fall back to real_test if no synthetic data
if not os.path.exists(TRAIN_IMAGE_DIR) or not os.listdir(TRAIN_IMAGE_DIR):
    TRAIN_IMAGE_DIR = os.path.join('dataset', 'real_test')
    LABELS_FILE = 'labels.json'
    print("  Using real_test/ for training (no synthetic data found)")

if not os.path.exists(VAL_IMAGE_DIR) or not os.listdir(VAL_IMAGE_DIR):
    VAL_IMAGE_DIR = TRAIN_IMAGE_DIR  # Use same for val if no separate set
    print("  Using training data for validation (no separate val set)")


def iou_score(pred, target, threshold=0.5):
    """Compute Intersection over Union."""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def train():
    print("=" * 60)
    print("  MotivAid — Training Segmentation Model")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    os.makedirs('models', exist_ok=True)

    # ─── Model ────────────────────────────────────────
    model = smp.Unet(
        'mobilenet_v2',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation='sigmoid'
    ).to(DEVICE)

    print(f"\n  Model: U-Net with MobileNetV2 encoder")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ─── Data ─────────────────────────────────────────
    train_ds = BloodLossDataset(TRAIN_IMAGE_DIR, MASK_DIR, LABELS_FILE, mode='train')
    val_ds = BloodLossDataset(VAL_IMAGE_DIR, MASK_DIR, LABELS_FILE, mode='val')

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if len(train_ds) == 0:
        print("\n  ERROR: No training data found!")
        print("  Add labeled images to dataset/ and re-run.")
        return

    # ─── Training setup ───────────────────────────────
    criterion = smp.losses.DiceLoss('binary')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5)

    best_iou = 0.0

    # ─── Training loop ────────────────────────────────
    print(f"\n  Training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_dl:
            images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)

            pred = model(images)
            loss = criterion(pred, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        # ─── Validation ──────────────────────────────
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for batch in val_dl:
                images = batch['image'].to(DEVICE)
                masks = batch['mask'].to(DEVICE)
                pred = model(images)
                val_iou += iou_score(pred, masks)

        val_iou /= len(val_dl)
        scheduler.step(train_loss)

        # Print progress
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | LR: {lr:.2e}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  >>> Saved best model (IoU: {best_iou:.4f})")

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Best IoU: {best_iou:.4f}")
    print(f"  Model saved to: {BEST_MODEL_PATH}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    train()

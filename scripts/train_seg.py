"""
MotivAid Blood Loss — Segmentation Model Training
===================================================
Trains a U-Net (MobileNetV2 backbone) to segment blood stains.
Run on Google Colab with GPU for faster training.

Expected output: models/seg_best.pt
"""

import os
import sys
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import BloodLossDataset

# -- Config --
EPOCHS = 80
BATCH_SIZE = 16
LR = 1e-4
PATIENCE = 10  # STOP after 10 epochs if no improvement
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define paths
TRAIN_IMAGE_DIR = os.path.join('dataset', 'synthetic_train')
VAL_IMAGE_DIR = os.path.join('dataset', 'synthetic_val')
MASK_DIR = os.path.join('dataset', 'masks')
LABELS_FILE = os.path.join('dataset', 'synthetic_labels.json')

def train():
    print("=" * 60)
    print("  MotivAid — Training Segmentation Model")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    os.makedirs('models', exist_ok=True)

    # -- Data (Now with a SEPARATE Validation Set) --
    train_ds = BloodLossDataset(TRAIN_IMAGE_DIR, MASK_DIR, LABELS_FILE, mode='train')
    val_ds   = BloodLossDataset(VAL_IMAGE_DIR, MASK_DIR, LABELS_FILE, mode='val')

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if len(train_ds) == 0:
        print("\n  ERROR: No training data found!")
        return

    # -- Model --
    model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid').to(DEVICE)
    # Using simple DiceLoss + BCELoss combo
    loss_fn = smp.losses.DiceLoss('binary') + smp.losses.SoftBCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # -- Training with Early Stopping --
    best_iou = 0.0
    patience_counter = 0
    best_model_path = os.path.join('models', 'seg_best.pt')

    print(f"\n  Training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_dl:
            images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)

            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        # Validation check
        model.eval()
        val_loss = 0.0
        intersection_sum = 0.0
        union_sum = 0.0

        with torch.no_grad():
            for batch in val_dl:
                images = batch['image'].to(DEVICE)
                masks = batch['mask'].to(DEVICE)

                preds = model(images)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()

                # Calculate IoU
                preds_bin = (preds > 0.5).float()
                intersection = (preds_bin * masks).sum()
                union = preds_bin.sum() + masks.sum() - intersection
                
                intersection_sum += intersection.item()
                union_sum += union.item()

        val_loss /= len(val_dl)
        val_iou = intersection_sum / (union_sum + 1e-6)

        print(f"  Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> Saved best model (IoU: {best_iou:.4f})")
            patience_counter = 0  # reset
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"\n  Stopping Early at Epoch {epoch}. Model has stopped improving.")
            break

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Best Val IoU: {best_iou:.4f}")
    print(f"  Model saved to: {best_model_path}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    train()

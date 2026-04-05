"""
MotivAid Blood Loss — Regression Model Training
=================================================
Trains a MobileNetV3 regression model to predict blood volume (mL)
from masked stain images.

Run AFTER train_seg.py has completed.
Run on Google Colab with GPU for faster training.

Expected output: models/reg_best.pt
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import BloodLossDataset
import segmentation_models_pytorch as smp

# ─── Config ───────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
EPOCHS = 80
LR = 5e-4
BEST_MODEL_PATH = os.path.join('models', 'reg_best.pt')
SEG_MODEL_PATH = os.path.join('models', 'seg_best.pt')

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
    VAL_IMAGE_DIR = TRAIN_IMAGE_DIR
    print("  Using training data for validation (no separate val set)")


class BloodLossRegressor(nn.Module):
    """
    Regression model for blood volume estimation.
    Uses MobileNetV3-Small backbone with additional surface-type
    and extra features as input.
    
    Input:  masked image (3, 224, 224) + surface one-hot (5) + extras (3)
    Output: log(mL) scalar
    """

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.features = backbone.features
        # 576 features from MobileNetV3-Small backbone
        # + 5 surface one-hot + 3 extra features = 584
        self.head = nn.Sequential(
            nn.Linear(576 + 5 + 3, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.Hardswish(),
            nn.Linear(64, 1)
        )

    def forward(self, image, surface_onehot, extras):
        x = self.features(image)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)  # (B, 576)
        x = torch.cat([x, surface_onehot, extras], dim=1)  # (B, 584)
        return self.head(x)


def train():
    print("=" * 60)
    print("  MotivAid — Training Regression Model")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    os.makedirs('models', exist_ok=True)

    # ─── Load segmentation model for masking ──────────
    seg_model = None
    if os.path.exists(SEG_MODEL_PATH):
        seg_model = smp.Unet(
            'mobilenet_v2', encoder_weights=None,
            in_channels=3, classes=1, activation='sigmoid'
        ).to(DEVICE)
        seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
        seg_model.eval()
        print(f"  Loaded segmentation model from {SEG_MODEL_PATH}")
    else:
        print(f"  WARNING: No segmentation model found at {SEG_MODEL_PATH}")
        print("  Using raw images without masking (less accurate)")

    # ─── Regression model ─────────────────────────────
    model = BloodLossRegressor().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: MobileNetV3-Small Regressor")
    print(f"  Parameters: {total_params:,}")

    # ─── Data ─────────────────────────────────────────
    train_ds = BloodLossDataset(TRAIN_IMAGE_DIR, MASK_DIR, LABELS_FILE, mode='train')
    val_ds = BloodLossDataset(VAL_IMAGE_DIR, MASK_DIR, LABELS_FILE, mode='val')

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if len(train_ds) == 0:
        print("\n  ERROR: No training data found!")
        return

    # ─── Training setup ───────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.5)

    best_mae = float('inf')

    # ─── Training loop ────────────────────────────────
    print(f"\n  Training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_dl:
            images = batch['image'].to(DEVICE)
            log_volumes = batch['log_volume'].to(DEVICE)
            surface_oh = batch['surface_onehot'].to(DEVICE)

            # Apply segmentation mask if model available
            if seg_model is not None:
                with torch.no_grad():
                    mask = (seg_model(images) > 0.5).float()
                    images = images * mask

            # Resize to 224 for MobileNetV3
            images = nn.functional.interpolate(images, size=224, mode='bilinear',
                                               align_corners=False)

            extras = torch.zeros(images.size(0), 3, device=DEVICE)
            pred_log = model(images, surface_oh, extras).squeeze(-1)

            loss = criterion(pred_log, log_volumes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        # ─── Validation ──────────────────────────────
        model.eval()
        val_errors = []
        with torch.no_grad():
            for batch in val_dl:
                images = batch['image'].to(DEVICE)
                true_ml = batch['volume_ml'].numpy()
                surface_oh = batch['surface_onehot'].to(DEVICE)

                if seg_model is not None:
                    mask = (seg_model(images) > 0.5).float()
                    images = images * mask

                images = nn.functional.interpolate(images, size=224, mode='bilinear',
                                                   align_corners=False)
                extras = torch.zeros(images.size(0), 3, device=DEVICE)
                pred_log = model(images, surface_oh, extras).squeeze(-1)
                pred_ml = torch.exp(pred_log).cpu().numpy()

                errors = np.abs(pred_ml - true_ml)
                val_errors.extend(errors.tolist())

        val_mae = np.mean(val_errors) if val_errors else float('inf')
        scheduler.step(val_mae)

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"Val MAE: {val_mae:.1f} mL | LR: {lr:.2e}")

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  >>> Saved best model (MAE: {best_mae:.1f} mL)")

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Best MAE: {best_mae:.1f} mL")
    print(f"  Model saved to: {BEST_MODEL_PATH}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    train()

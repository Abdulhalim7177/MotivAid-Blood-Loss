"""
MotivAid Blood Loss — Model Evaluation
========================================
Evaluates both models (segmentation + regression) on real test images.
Reports per-image predictions and overall MAE accuracy.

Usage:  python scripts/evaluate.py
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.models as models
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import get_transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SURFACE_MAP = {'pad': 0, 'gauze': 1, 'sheet': 2, 'drape': 3, 'other': 4}


class BloodLossRegressor(nn.Module):
    """Same architecture as in train_reg.py."""

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.features = backbone.features
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
        x = x.flatten(1)
        x = torch.cat([x, surface_onehot, extras], dim=1)
        return self.head(x)


def main():
    print("=" * 60)
    print("  MotivAid — Model Evaluation")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ─── Load models ──────────────────────────────────
    seg_path = os.path.join('models', 'seg_best.pt')
    reg_path = os.path.join('models', 'reg_best.pt')

    if not os.path.exists(seg_path):
        print(f"\n  ERROR: Segmentation model not found at {seg_path}")
        print("  Run train_seg.py first.")
        return

    if not os.path.exists(reg_path):
        print(f"\n  ERROR: Regression model not found at {reg_path}")
        print("  Run train_reg.py first.")
        return

    # Load segmentation model
    seg_model = smp.Unet(
        'mobilenet_v2', encoder_weights=None,
        in_channels=3, classes=1, activation='sigmoid'
    ).to(DEVICE)
    seg_model.load_state_dict(torch.load(seg_path, map_location=DEVICE))
    seg_model.eval()
    print("  ✓ Segmentation model loaded")

    # Load regression model
    reg_model = BloodLossRegressor().to(DEVICE)
    reg_model.load_state_dict(torch.load(reg_path, map_location=DEVICE))
    reg_model.eval()
    print("  ✓ Regression model loaded")

    # ─── Load labels ──────────────────────────────────
    labels_file = 'labels.json'
    if not os.path.exists(labels_file):
        labels_file = os.path.join('dataset', 'synthetic_labels.json')
    
    with open(labels_file) as f:
        all_labels = json.load(f)
    
    # Get real_test labels
    labels = all_labels.get('real_test', {})
    if not labels:
        # Try to use any available labels
        for split_data in all_labels.values():
            if isinstance(split_data, dict):
                labels.update(split_data)

    if not labels:
        print("\n  ERROR: No labels found. Run label_images.py first.")
        return

    print(f"  Evaluating on {len(labels)} images...\n")

    # ─── Evaluate ─────────────────────────────────────
    transforms = get_transforms('val')
    errors = []

    print(f"  {'File':<35} {'True':>6} {'Pred':>6} {'Error':>6} {'%Err':>5}")
    print(f"  {'─' * 35} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 5}")

    for fname, info in labels.items():
        path = os.path.join('dataset', 'real_test', fname)
        if not os.path.exists(path):
            continue

        img_np = np.array(Image.open(path).convert('RGB'))
        aug = transforms(image=img_np)
        img_t = aug['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Segmentation
            mask = (seg_model(img_t) > 0.5).float()
            masked = img_t * mask

            # Resize for regression
            masked_224 = nn.functional.interpolate(
                masked, size=224, mode='bilinear', align_corners=False)

            # Surface type
            s_idx = SURFACE_MAP.get(info.get('surface_type', 'other'), 4)
            s_oh = torch.zeros(1, 5, device=DEVICE)
            s_oh[0, s_idx] = 1.0
            extras = torch.zeros(1, 3, device=DEVICE)

            # Prediction
            log_pred = reg_model(masked_224, s_oh, extras)
            pred_ml = torch.exp(log_pred).item()

        true_ml = info['volume_ml']
        err = abs(pred_ml - true_ml)
        pct_err = err / max(true_ml, 1) * 100

        errors.append({
            'file': fname,
            'true': true_ml,
            'pred': round(pred_ml, 1),
            'error': round(err, 1),
            'pct': round(pct_err, 1)
        })

        print(f"  {fname:<35} {true_ml:>5}  {pred_ml:>5.1f}  {err:>5.1f}  {pct_err:>4.0f}%")

    # ─── Summary ──────────────────────────────────────
    if errors:
        mae = np.mean([e['error'] for e in errors])
        mean_pct = np.mean([e['pct'] for e in errors])
        within_15 = sum(1 for e in errors if e['pct'] <= 15) / len(errors) * 100
        within_30 = sum(1 for e in errors if e['pct'] <= 30) / len(errors) * 100

        print(f"\n  {'─' * 60}")
        print(f"  Results Summary:")
        print(f"    MAE:              {mae:.1f} mL")
        print(f"    Mean % error:     {mean_pct:.1f}%")
        print(f"    Within 15% error: {within_15:.0f}%")
        print(f"    Within 30% error: {within_30:.0f}%")
        print(f"  {'─' * 60}")

        if mae < 50:
            print("  ✓ GOOD — MAE under 50 mL. Ready for prototype.")
        elif mae < 70:
            print("  ~ ACCEPTABLE — MAE 50-70 mL. Usable for first prototype.")
        else:
            print("  ✗ NEEDS IMPROVEMENT — MAE above 70 mL. Need more/better data.")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()

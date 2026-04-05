"""
MotivAid Blood Loss — ONNX Export
==================================
Exports trained PyTorch models to ONNX format.
ONNX is the bridge format between PyTorch and TFLite.

Usage:  python scripts/export_onnx.py

Expected output:
  - models/seg_model.onnx
  - models/reg_model.onnx
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp

DEVICE = 'cpu'  # Export on CPU for compatibility


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


def export_segmentation():
    """Export segmentation model to ONNX."""
    seg_path = os.path.join('models', 'seg_best.pt')
    if not os.path.exists(seg_path):
        print(f"  ✗ Segmentation model not found at {seg_path}")
        return False

    model = smp.Unet(
        'mobilenet_v2', encoder_weights=None,
        in_channels=3, classes=1, activation='sigmoid'
    ).to(DEVICE)
    model.load_state_dict(torch.load(seg_path, map_location=DEVICE))
    model.eval()

    # Dummy input: batch of 1, 3 channels, 256x256
    dummy_input = torch.randn(1, 3, 256, 256, device=DEVICE)
    output_path = os.path.join('models', 'seg_model.onnx')

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['mask'],
        dynamic_axes={
            'image': {0: 'batch'},
            'mask': {0: 'batch'}
        },
        opset_version=13
    )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✓ Segmentation model exported: {output_path} ({size_mb:.2f} MB)")
    return True


def export_regression():
    """Export regression model to ONNX."""
    reg_path = os.path.join('models', 'reg_best.pt')
    if not os.path.exists(reg_path):
        print(f"  ✗ Regression model not found at {reg_path}")
        return False

    model = BloodLossRegressor().to(DEVICE)
    model.load_state_dict(torch.load(reg_path, map_location=DEVICE))
    model.eval()

    # Dummy inputs
    dummy_image = torch.randn(1, 3, 224, 224, device=DEVICE)
    dummy_surface = torch.zeros(1, 5, device=DEVICE)
    dummy_extras = torch.zeros(1, 3, device=DEVICE)
    output_path = os.path.join('models', 'reg_model.onnx')

    torch.onnx.export(
        model,
        (dummy_image, dummy_surface, dummy_extras),
        output_path,
        input_names=['image', 'surface_type', 'extra_features'],
        output_names=['log_ml'],
        dynamic_axes={
            'image': {0: 'batch'},
            'surface_type': {0: 'batch'},
            'extra_features': {0: 'batch'},
            'log_ml': {0: 'batch'}
        },
        opset_version=13
    )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✓ Regression model exported: {output_path} ({size_mb:.2f} MB)")
    return True


def main():
    print("=" * 60)
    print("  MotivAid — Exporting Models to ONNX")
    print("=" * 60)

    os.makedirs('models', exist_ok=True)

    seg_ok = export_segmentation()
    reg_ok = export_regression()

    print(f"\n  Summary:")
    print(f"    Segmentation: {'✓ Exported' if seg_ok else '✗ Failed'}")
    print(f"    Regression:   {'✓ Exported' if reg_ok else '✗ Failed'}")

    if seg_ok and reg_ok:
        print(f"\n  Next steps:")
        print(f"    1. Convert to TFLite (on Colab):")
        print(f"       !pip install onnx2tf tensorflow")
        print(f"       !onnx2tf -i models/seg_model.onnx -o models/seg_tf/")
        print(f"       !onnx2tf -i models/reg_model.onnx -o models/reg_tf/")
        print(f"    2. Or run the MVP prototype:")
        print(f"       python mvp_app/app.py")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()

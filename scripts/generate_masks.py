"""
MotivAid Blood Loss — Mask Generator
=====================================
Generates approximate segmentation masks using colour detection (HSV thresholding).
These masks indicate which pixels are blood-stained.

Usage:  python scripts/generate_masks.py
"""

import cv2
import os
import json
import numpy as np
from PIL import Image

IMAGE_DIR = os.path.join('dataset', 'real_test')
MASK_DIR = os.path.join('dataset', 'masks')
LABELS_FILE = 'labels.json'

os.makedirs(MASK_DIR, exist_ok=True)


def make_mask(img_path, surface_type='pad'):
    """
    Generate a blood stain mask using HSV colour detection.
    Blood appears in red/dark-red range in HSV space.
    Different surfaces may need slightly different thresholds.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  WARNING: Could not read {img_path}")
        return None

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Blood detection in HSV space
    # H = hue (0-180 in OpenCV), S = saturation, V = value
    # Blood is in the red range: H ~0-15 and H ~160-180

    # Lower red range
    mask1 = cv2.inRange(hsv, (0, 60, 40), (15, 255, 255))
    # Upper red range
    mask2 = cv2.inRange(hsv, (160, 60, 40), (180, 255, 255))
    # Dark blood (very low saturation but dark value)
    mask3 = cv2.inRange(hsv, (0, 30, 20), (20, 255, 120))

    mask = mask1 | mask2 | mask3

    # Surface-specific adjustments
    if surface_type in ('drape', 'sheet'):
        # Surgical drapes are often blue/green — blood appears darker
        # Expand to include brown tones
        mask_brown = cv2.inRange(hsv, (5, 40, 30), (25, 255, 200))
        mask = mask | mask_brown

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove small noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = mask.shape[0] * mask.shape[1] * 0.001  # At least 0.1% of image
    clean_mask = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean_mask, [c], -1, 255, -1)

    return clean_mask


def main():
    print("=" * 60)
    print("  MotivAid — Generating Segmentation Masks")
    print("=" * 60)

    # Load labels to get surface types
    surface_map = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            labels = json.load(f)
        for fname, info in labels.get('real_test', {}).items():
            surface_map[fname] = info.get('surface_type', 'pad')
        print(f"  Loaded labels for {len(surface_map)} images")

    # Process all images
    image_files = sorted([f for f in os.listdir(IMAGE_DIR)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not image_files:
        print(f"\n  No images found in {IMAGE_DIR}/")
        return

    print(f"  Processing {len(image_files)} images...\n")

    generated = 0
    for fname in image_files:
        path = os.path.join(IMAGE_DIR, fname)
        surface = surface_map.get(fname, 'pad')

        mask = make_mask(path, surface)
        if mask is not None:
            # Save mask with same name
            mask_name = os.path.splitext(fname)[0] + '_mask.png'
            mask_path = os.path.join(MASK_DIR, mask_name)
            cv2.imwrite(mask_path, mask)

            # Report coverage
            coverage = mask.sum() / (mask.shape[0] * mask.shape[1] * 255) * 100
            print(f"  ✓ {fname} → {mask_name}  (stain coverage: {coverage:.1f}%)")
            generated += 1
        else:
            print(f"  ✗ {fname} — failed")

    print(f"\n  Generated {generated} masks in {MASK_DIR}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

"""
MotivAid Blood Loss — Image Auditor
====================================
Scans dataset/real_test/ to verify images are valid and reports stats.

Usage:  python scripts/audit_images.py
"""

import os
import json
from PIL import Image

IMAGE_DIR = os.path.join('dataset', 'real_test')
LABELS_FILE = 'labels.json'
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
SURFACES = ['pad', 'gauze', 'sheet', 'drape', 'other']


def main():
    print("=" * 60)
    print("  MotivAid — Image Audit Report")
    print("=" * 60)

    if not os.path.exists(IMAGE_DIR):
        print(f"\n  ERROR: {IMAGE_DIR}/ not found.")
        print("  Create it and add your images first.")
        return

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
    print(f"\n  Found {len(files)} image files in {IMAGE_DIR}/")

    if not files:
        print("  No images to audit. Add images and re-run.")
        return

    # Check each image
    valid = []
    invalid = []
    sizes = []
    for fname in sorted(files):
        path = os.path.join(IMAGE_DIR, fname)
        try:
            img = Image.open(path)
            img.verify()  # Verify it's a valid image
            # Re-open to get size (verify closes the file)
            img = Image.open(path)
            w, h = img.size
            sizes.append((w, h))
            valid.append(fname)
        except Exception as e:
            invalid.append((fname, str(e)))

    print(f"  Valid images:   {len(valid)}")
    print(f"  Invalid images: {len(invalid)}")

    if invalid:
        print("\n  ⚠ Invalid images:")
        for fname, err in invalid:
            print(f"    - {fname}: {err}")

    if valid:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"\n  Size range: {min(widths)}x{min(heights)} — {max(widths)}x{max(heights)}")

    # Check for labels
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            labels = json.load(f)
        labeled = labels.get('real_test', {})
        print(f"\n  Labels file found: {len(labeled)} images labeled")
        unlabeled = [f for f in valid if f not in labeled]
        if unlabeled:
            print(f"  ⚠ {len(unlabeled)} images are NOT labeled:")
            for f in unlabeled[:10]:
                print(f"    - {f}")
            if len(unlabeled) > 10:
                print(f"    ... and {len(unlabeled) - 10} more")
        else:
            print("  ✓ All valid images are labeled!")

        # Surface type distribution
        surface_counts = {}
        for info in labeled.values():
            s = info.get('surface_type', 'unknown')
            surface_counts[s] = surface_counts.get(s, 0) + 1
        if surface_counts:
            print(f"\n  Surface type distribution:")
            for s, c in sorted(surface_counts.items()):
                print(f"    {s}: {c} images")

        # Volume distribution
        volumes = [info['volume_ml'] for info in labeled.values() if 'volume_ml' in info]
        if volumes:
            print(f"\n  Volume range: {min(volumes)} — {max(volumes)} mL")
            print(f"  Average volume: {sum(volumes) / len(volumes):.0f} mL")
    else:
        print(f"\n  ⚠ No labels.json found.")
        print(f"  Run: python scripts/label_images.py  to label your images.")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()

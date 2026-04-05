"""
MotivAid Blood Loss — Image Labeling Tool
==========================================
Usage:
  1. Put your unlabeled images into dataset/real_test/
  2. Run:  python scripts/label_images.py
  3. For each image, you'll see it displayed and be prompted to enter:
       - Surface type (pad / gauze / sheet / drape / other)
       - Estimated blood volume in mL
  4. Results are saved to labels.json

The tool also renames images to the convention: {surface}_{volume}mL_{index}.jpg
"""

import os
import json
import sys
import shutil
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt

IMAGE_DIR = os.path.join('dataset', 'real_test')
LABELS_FILE = 'labels.json'
VALID_SURFACES = ['pad', 'gauze', 'sheet', 'drape', 'other']
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def load_existing_labels():
    """Load existing labels if they exist."""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            return json.load(f)
    return {'real_test': {}}


def save_labels(labels):
    """Save labels to JSON file."""
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"\n  Labels saved to {LABELS_FILE}")


def get_image_files(directory):
    """Get all image files in directory."""
    files = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(VALID_EXTENSIONS):
            files.append(fname)
    return files


def show_image(path, title=""):
    """Display an image using matplotlib."""
    img = Image.open(path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


def get_surface_type():
    """Prompt user for surface type."""
    print(f"\n  Surface types: {', '.join(VALID_SURFACES)}")
    while True:
        surface = input("  Enter surface type: ").strip().lower()
        if surface in VALID_SURFACES:
            return surface
        print(f"  Invalid. Choose from: {', '.join(VALID_SURFACES)}")


def get_volume():
    """Prompt user for blood volume in mL."""
    while True:
        try:
            vol = input("  Enter estimated blood volume (mL): ").strip()
            vol = int(vol)
            if 0 <= vol <= 5000:
                return vol
            print("  Volume should be between 0 and 5000 mL.")
        except ValueError:
            print("  Please enter a whole number (e.g. 50, 100, 250).")


def main():
    print("=" * 60)
    print("  MotivAid Blood Loss — Image Labeling Tool")
    print("=" * 60)

    # Check for images
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        print(f"\n  Created {IMAGE_DIR}/ — please add your images there and re-run.")
        return

    images = get_image_files(IMAGE_DIR)
    if not images:
        print(f"\n  No images found in {IMAGE_DIR}/")
        print("  Please add your images (.jpg, .png, etc.) and re-run.")
        return

    print(f"\n  Found {len(images)} images in {IMAGE_DIR}/")
    labels = load_existing_labels()
    already_labeled = set(labels.get('real_test', {}).keys())

    # Filter out already-labeled images
    unlabeled = [f for f in images if f not in already_labeled]
    if not unlabeled:
        print("  All images are already labeled!")
        print(f"  Labels file: {LABELS_FILE}")
        print(f"  To re-label, delete {LABELS_FILE} and re-run.")
        return

    print(f"  {len(unlabeled)} unlabeled images to process.")
    print(f"  ({len(already_labeled)} already labeled)")
    print("\n  For each image, enter the surface type and estimated volume.")
    print("  Type 'skip' to skip an image, 'quit' to save and exit.\n")

    counters = {}  # Track per-surface count for renaming
    for surface in VALID_SURFACES:
        existing = [f for f in os.listdir(IMAGE_DIR) if f.startswith(surface + '_')]
        counters[surface] = len(existing) + 1

    labeled_count = 0
    for i, fname in enumerate(unlabeled):
        path = os.path.join(IMAGE_DIR, fname)
        print(f"\n  --- Image {i+1}/{len(unlabeled)}: {fname} ---")

        # Show the image
        try:
            show_image(path, title=f"Image {i+1}/{len(unlabeled)}: {fname}")
        except Exception as e:
            print(f"  Could not display image: {e}")
            print("  (You can still label it based on the filename)")

        # Get user action
        action = input("\n  Label this image? (yes/skip/quit): ").strip().lower()
        if action == 'quit':
            break
        if action == 'skip':
            plt.close('all')
            continue

        # Get labels
        surface = get_surface_type()
        volume = get_volume()

        # Rename file to convention: {surface}_{volume}mL_{index}.jpg
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ('.jpg', '.jpeg'):
            ext = '.jpg'  # Standardize to jpg
        new_name = f"{surface}_{volume:03d}mL_{counters[surface]:03d}{ext}"
        counters[surface] += 1

        new_path = os.path.join(IMAGE_DIR, new_name)
        if fname != new_name:
            # Keep original, copy with new name
            if not os.path.exists(new_path):
                shutil.copy2(path, new_path)
                print(f"  Copied as: {new_name}")
            else:
                new_name = fname  # Keep original if conflict
                print(f"  Keeping original name: {fname}")
        else:
            print(f"  Already named correctly: {fname}")

        # Store label
        labels['real_test'][new_name] = {
            'volume_ml': volume,
            'surface_type': surface,
            'original_name': fname
        }
        labeled_count += 1

        plt.close('all')
        print(f"  ✓ Labeled: {new_name} → {surface}, {volume} mL")

    # Save
    save_labels(labels)
    print(f"\n{'=' * 60}")
    print(f"  Done! Labeled {labeled_count} images.")
    print(f"  Labels saved to: {LABELS_FILE}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

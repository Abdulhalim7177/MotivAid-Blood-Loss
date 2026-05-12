"""
MotivAid Blood Loss — Label Builder
=====================================
Builds synthetic_labels.json from filenames in dataset/synthetic_train/.
Filename convention: {surface}_{volume}mL_{index}.jpg

Also works with labels.json from the labeling tool.

Usage:  python scripts/build_labels.py
"""

import os
import re
import json

SURFACES = ['bowl', 'container', 'pad', 'pampers', 'drape', 'floor', 'cloth', 'bedsheet', 'towel', 'gauze', 'sheet', 'floor-and-cloth', 'cloth-and-floor', 'pad-and-container', 'pad-and-floor', 'other']
DIRS = {
    'synthetic_train': os.path.join('dataset', 'synthetic_train'),
    'synthetic_val': os.path.join('dataset', 'synthetic_val'),
    'real_test': os.path.join('dataset', 'real_test'),
}
OUTPUT_FILE = os.path.join('dataset', 'synthetic_labels.json')
LABELS_FILE = 'labels.json'


def parse_filename(fname):
    """
    Parse volume, surface type, and metadata from filename.
    """
    # Try to extract volume
    m = re.search(r'(\d+)\s*mL', fname, re.IGNORECASE)
    if not m:
        return None, None, None

    volume = int(m.group(1))

    # Try to extract surface type
    surface = 'other'
    fname_lower = fname.lower()
    for s in SURFACES:
        if fname_lower.startswith(s):
            surface = s
            break

    # Extract metadata
    # Distance
    dist_m = re.search(r'(\d+)\s*cm', fname, re.IGNORECASE)
    distance = int(dist_m.group(1)) if dist_m else 40  # default 40cm

    # Lighting
    lighting = 'daylight'
    if 'led' in fname_lower:
        lighting = 'led'
    elif 'dim' in fname_lower:
        lighting = 'dim'

    # Clot
    clot_m = re.search(r'clot-(yes|no)', fname_lower)
    has_clot = (clot_m.group(1) == 'yes') if clot_m else False

    metadata = {
        'distance_cm': distance,
        'lighting': lighting,
        'has_clot': has_clot
    }

    return volume, surface, metadata


def main():
    print("=" * 60)
    print("  MotivAid — Building Labels File")
    print("=" * 60)

    all_labels = {}

    for split_name, dir_path in DIRS.items():
        if not os.path.exists(dir_path):
            print(f"\n  {split_name}/: directory not found, skipping")
            continue

        files = [f for f in sorted(os.listdir(dir_path))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not files:
            print(f"\n  {split_name}/: no images found")
            continue

        print(f"\n  {split_name}/: {len(files)} images")
        split_labels = {}
        parsed = 0
        failed = 0

        for fname in files:
            volume, surface, metadata = parse_filename(fname)
            if volume is not None:
                if volume > 3000:
                    print(f"    WARNING: skipping outlier {fname} ({volume} mL > 3000mL)")
                    failed += 1
                    continue
                split_labels[fname] = {
                    'volume_ml': volume,
                    'surface_type': surface,
                    **metadata
                }
                parsed += 1
            else:
                print(f"    WARNING: could not parse volume from {fname}")
                failed += 1

        all_labels[split_name] = split_labels
        print(f"    Parsed: {parsed}, Failed: {failed}")

    # Merge with existing labels.json if present
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            manual_labels = json.load(f)
        print(f"\n  Merging with existing {LABELS_FILE}...")
        for split_name, entries in manual_labels.items():
            if split_name not in all_labels:
                all_labels[split_name] = {}
            for fname, info in entries.items():
                if fname not in all_labels[split_name]:
                    all_labels[split_name][fname] = info
                    
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_labels, f, indent=2)

    total = sum(len(v) for v in all_labels.values())
    print(f"\n  Saved {total} labels to {OUTPUT_FILE}")

    # Also update labels.json
    with open(LABELS_FILE, 'w') as f:
        json.dump(all_labels, f, indent=2)
    print(f"  Also updated {LABELS_FILE}")

    # Summary
    for split_name, entries in all_labels.items():
        if entries:
            volumes = [e['volume_ml'] for e in entries.values()]
            print(f"\n  {split_name}:")
            print(f"    Count: {len(entries)}")
            print(f"    Volume range: {min(volumes)} — {max(volumes)} mL")
            print(f"    Average: {sum(volumes)/len(volumes):.0f} mL")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()

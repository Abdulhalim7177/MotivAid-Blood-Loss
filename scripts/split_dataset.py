import os
import shutil
import re
import random
from collections import defaultdict

random.seed(42)

def parse_features(fname):
    """Parse key physical features to use as a stratification key."""
    # Volume
    m = re.search(r'(\d+)\s*mL', fname, re.IGNORECASE)
    volume = int(m.group(1)) if m else 0

    # Surface
    surface = 'other'
    fname_lower = fname.lower()
    for s in ['bowl', 'container', 'pad', 'pampers', 'drape', 'floor-and-cloth', 'cloth-and-floor', 'pad-and-container', 'pad-and-floor', 'floor', 'cloth', 'bedsheet', 'towel', 'gauze', 'sheet']:
        if fname_lower.startswith(s):
            surface = s
            break

    # Lighting
    lighting = 'daylight'
    if 'led' in fname_lower: lighting = 'led'
    elif 'dim' in fname_lower: lighting = 'dim'

    # Clot
    clot_m = re.search(r'clot-(yes|no)', fname_lower)
    has_clot = (clot_m.group(1) == 'yes') if clot_m else False
    
    return f"{surface}_{volume}_{lighting}_{has_clot}"

def main():
    print("=" * 60)
    print("  MotivAid — Stratified Dataset Splitter & Renamer")
    print("=" * 60)

    dataset_dir = 'dataset'
    splits = ['synthetic_train', 'synthetic_val', 'synthetic_test']
    
    # 1. Gather all files from all split directories
    all_files = []
    for split in splits:
        d = os.path.join(dataset_dir, split)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(d, f))

    if not all_files:
        print("  ERROR: No images found.")
        return

    print(f"  Found {len(all_files)} total images to process.")

    # 2. Extract base names, assign indices, and group by features
    base_counts = defaultdict(int)
    renamed_files = [] # list of (old_path, new_name, stratification_key)
    
    for path in all_files:
        fname = os.path.basename(path)
        
        # Remove WhatsApp_Image and everything after it
        m = re.match(r'(.*?)_WhatsApp_Image', fname, re.IGNORECASE)
        if m:
            base_name = m.group(1)
        else:
            base_name = os.path.splitext(fname)[0]
            # Strip trailing duplicate numbers like __1_ or _2
            base_name = re.sub(r'__\d+_$', '', base_name)
            base_name = re.sub(r'_\d+$', '', base_name)
            
        ext = os.path.splitext(fname)[1]
        
        # Clean up double underscores
        base_name = re.sub(r'_+', '_', base_name).strip('_')
        
        # Keep track of counts to assign a unique _001, _002 to the end
        base_counts[base_name] += 1
        new_name = f"{base_name}_{base_counts[base_name]:03d}{ext}"
        
        # Calculate stratify key
        strat_key = parse_features(base_name)
        renamed_files.append((path, new_name, strat_key))

    # 3. Stratified Splitting
    stratified_groups = defaultdict(list)
    for item in renamed_files:
        stratified_groups[item[2]].append(item)
        
    print(f"  Identified {len(stratified_groups)} unique feature profiles for stratification.")
        
    train_files = []
    val_files = []
    test_files = []
    
    for key, items in stratified_groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        # Handle small groups gracefully
        if n == 1:
            train_files.extend(items)
        elif n == 2:
            train_files.append(items[0])
            val_files.append(items[1])
        else:
            train_files.extend(items[:n_train])
            val_files.extend(items[n_train:n_train+n_val])
            test_files.extend(items[n_train+n_val:])
            
    # 4. Move files securely via a temporary directory to avoid overwrite collisions
    temp_dir = os.path.join(dataset_dir, 'temp_renaming')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_train = []
    for old_path, new_name, _ in train_files:
        temp_path = os.path.join(temp_dir, new_name)
        shutil.move(old_path, temp_path)
        temp_train.append((temp_path, os.path.join(dataset_dir, 'synthetic_train', new_name)))
        
    temp_val = []
    for old_path, new_name, _ in val_files:
        temp_path = os.path.join(temp_dir, new_name)
        shutil.move(old_path, temp_path)
        temp_val.append((temp_path, os.path.join(dataset_dir, 'synthetic_val', new_name)))
        
    temp_test = []
    for old_path, new_name, _ in test_files:
        temp_path = os.path.join(temp_dir, new_name)
        shutil.move(old_path, temp_path)
        temp_test.append((temp_path, os.path.join(dataset_dir, 'synthetic_test', new_name)))
        
    # Ensure final directories exist
    for split in splits:
        os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)
        
    # Move from temp to final
    for src, dst in temp_train: shutil.move(src, dst)
    for src, dst in temp_val: shutil.move(src, dst)
    for src, dst in temp_test: shutil.move(src, dst)
    
    shutil.rmtree(temp_dir)
    
    print("\n  Final Renamed & Split Counts:")
    print(f"    Train: {len(temp_train)} images ({len(temp_train)/len(all_files)*100:.1f}%)")
    print(f"    Val:   {len(temp_val)} images ({len(temp_val)/len(all_files)*100:.1f}%)")
    print(f"    Test:  {len(temp_test)} images ({len(temp_test)/len(all_files)*100:.1f}%)")
    print("\n  Sample renamed file: " + (temp_train[0][1] if temp_train else "N/A"))
    print("\n  Done! Next, please run: python scripts/build_labels.py")
    print("=" * 60)

if __name__ == '__main__':
    main()

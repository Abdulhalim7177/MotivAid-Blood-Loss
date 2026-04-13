import os
import hashlib
import re
from collections import defaultdict

def get_file_hash(file_path):
    """Calculate the MD5 hash of a file. Only identical files have the same hash."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def deduplicate_by_name_and_hash(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Scanning {len(files)} images for WhatsApp/System duplicates...")

    # Regex to find patterns like "Name (1).jpg" or "Name (2).jpg"
    dup_pattern = re.compile(r"^(.*)\s\(\d+\)(\.[^.]+)$")

    processed_hashes = {} # hash -> original_filename
    duplicates_removed = 0

    # First pass: Identify "Originals" (files WITHOUT the (1) suffix)
    # This ensures we have a reference to compare the (1) files against.
    for filename in files:
        if not dup_pattern.match(filename):
            path = os.path.join(directory, filename)
            f_hash = get_file_hash(path)
            processed_hashes[f_hash] = filename

    # Second pass: Check files WITH the (1) suffix
    for filename in files:
        match = dup_pattern.match(filename)
        if match:
            path = os.path.join(directory, filename)
            f_hash = get_file_hash(path)

            # If this hash already exists (meaning we have the original or another copy)
            if f_hash in processed_hashes:
                original = processed_hashes[f_hash]
                print(f"[!] Found exact duplicate: '{filename}' is identical to '{original}'")
                try:
                    os.remove(path)
                    print(f"    ✓ Deleted: {filename}")
                    duplicates_removed += 1
                except Exception as e:
                    print(f"    x Error deleting {filename}: {e}")
            else:
                # If it has (1) but we DON'T have a matching hash yet, 
                # treat this as a new unique file and keep it.
                processed_hashes[f_hash] = filename

    if duplicates_removed == 0:
        print("\nNo exact duplicates with (n) suffixes found.")
    else:
        print(f"\nClean up complete. Total duplicates removed: {duplicates_removed}")

if __name__ == "__main__":
    deduplicate_by_name_and_hash("blood-images")

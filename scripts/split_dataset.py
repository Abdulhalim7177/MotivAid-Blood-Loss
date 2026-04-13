import os
import random
import shutil

def split_data(train_dir, val_dir, split_ratio=0.2):
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Check if val is already populated
    if len(os.listdir(val_dir)) > 0:
        print("Validation directory is not empty. Skipping split.")
        return

    num_val = int(len(files) * split_ratio)
    val_files = random.sample(files, num_val)
    
    print(f"Moving {len(val_files)} images from Train to Val...")
    
    for f in val_files:
        shutil.move(os.path.join(train_dir, f), os.path.join(val_dir, f))
    
    print("Done.")

if __name__ == "__main__":
    split_data("dataset/synthetic_train", "dataset/synthetic_val")

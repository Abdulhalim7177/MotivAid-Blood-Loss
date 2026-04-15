import os
import shutil
import re
import cv2
import numpy as np
from PIL import Image, ImageOps
import platform

# --- Configuration ---
RAW_DIR = "blood-images"
DEST_DIR = "dataset/synthetic_train"

SURFACES = ["bowl", "container", "pad", "pampers", "drape", "floor", "bedsheet", "towel", "other"]
DISTANCES = ["20cm", "40cm"]
LIGHTING = ["daylight", "led", "other"]
CLOT_OPTIONS = ["yes", "no"]

os.makedirs(DEST_DIR, exist_ok=True)

def show_image_cv(image_path):
    """Open the image using OpenCV for auto-closing capability."""
    try:
        # Load with PIL first to handle EXIF rotation (phone photos)
        img_pil = Image.open(image_path)
        img_pil = ImageOps.exif_transpose(img_pil)
        
        # Convert to OpenCV format (BGR)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Resize to fit screen (max 800px)
        h, w = img_cv.shape[:2]
        max_dim = 800
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))
            
        cv2.imshow("Labeler - View image, then type in terminal", img_cv)
        cv2.waitKey(1)  # Refresh window
        return True
    except Exception as e:
        print(f"Error displaying image: {e}")
        return False

def get_choice(options, prompt):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
        # Keep CV window responsive
        cv2.waitKey(1) 
        try:
            choice = input("Enter number (or type 'other' for custom): ").strip().lower()
            if choice == 'other':
                return input("Enter custom value: ").strip().replace(" ", "-")
            
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected = options[idx]
                if selected == "other":
                    return input("Enter custom surface type: ").strip().replace(" ", "-")
                return selected
        except ValueError:
            pass
        print("Invalid choice, please try again.")

def main():
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print(f"No images found in {RAW_DIR}")
        return

    print(f"Found {len(files)} images to process.")
    
    for filename in files:
        print(f"\n" + "="*40)
        print(f"Processing: {filename}")
        
        src_path = os.path.join(RAW_DIR, filename)
        
        # Open image
        has_window = show_image_cv(src_path)
        
        # 1. Volume
        while True:
            cv2.waitKey(1)
            vol = input("Enter blood volume in mL (e.g., 50) or 'skip': ").strip().lower()
            if vol == 'skip': break
            if vol.isdigit(): break
            print("Please enter a number.")
        
        if vol == 'skip':
            print(f"Skipped {filename}")
            if has_window: cv2.destroyAllWindows()
            continue
            
        # 2. Surface
        surface = get_choice(SURFACES, "Select Surface Type:")
        
        # 3. Distance
        distance = get_choice(DISTANCES, "Select Distance:")
        
        # 4. Lighting
        light = get_choice(LIGHTING, "Select Lighting:")
        
        # 5. Clot
        clot = get_choice(CLOT_OPTIONS, "Is there a blood clot?")
        
        # Close the window immediately after labeling is done
        if has_window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
        # Create new filename
        clean_name = re.sub(r'[^a-zA-Z0-9.-]', '_', filename)
        new_filename = f"{surface}_{vol}mL_{distance}_{light}_clot-{clot}_{clean_name}"
        
        dst_path = os.path.join(DEST_DIR, new_filename)
        shutil.move(src_path, dst_path)
        print(f"\n✓ Moved and renamed to: {new_filename}")

    print("\n" + "="*40)
    print("Done! All images processed.")

if __name__ == "__main__":
    main()

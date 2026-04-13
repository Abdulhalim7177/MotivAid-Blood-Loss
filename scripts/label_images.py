import os
import shutil
import re

# --- Configuration ---
RAW_DIR = "blood-images"
DEST_DIR = "dataset/synthetic_train"

SURFACES = ["bowl", "container", "pad", "pampers", "drape", "floor", "bedsheet", "towel", "other"]
DISTANCES = ["20cm", "40cm"]
LIGHTING = ["daylight", "led", "other"]
CLOT_OPTIONS = ["yes", "no"]

os.makedirs(DEST_DIR, exist_ok=True)

def get_choice(options, prompt):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
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
        
        # 1. Volume
        while True:
            vol = input("Enter blood volume in mL (e.g., 50): ").strip()
            if vol.isdigit():
                break
            print("Please enter a number.")
            
        # 2. Surface
        surface = get_choice(SURFACES, "Select Surface Type:")
        
        # 3. Distance
        distance = get_choice(DISTANCES, "Select Distance:")
        
        # 4. Lighting
        light = get_choice(LIGHTING, "Select Lighting:")
        
        # 5. Clot
        clot = get_choice(CLOT_OPTIONS, "Is there a blood clot?")
        
        # Create new filename
        # Format: {surface}_{volume}mL_{distance}_{lighting}_clot-{clot}_{original_name}
        clean_name = re.sub(r'[^a-zA-Z0-9.-]', '_', filename)
        new_filename = f"{surface}_{vol}mL_{distance}_{light}_clot-{clot}_{clean_name}"
        
        src_path = os.path.join(RAW_DIR, filename)
        dst_path = os.path.join(DEST_DIR, new_filename)
        
        shutil.move(src_path, dst_path)
        print(f"\n✓ Moved and renamed to: {new_filename}")

    print("\n" + "="*40)
    print("Done! All images processed.")

if __name__ == "__main__":
    main()

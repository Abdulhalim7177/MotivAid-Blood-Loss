# MotivAid Blood Loss — Terminal Commands Reference

This document lists all the commands needed to run, train, and evaluate the blood loss AI models.

## 🚀 Getting Started

### 1. Environment Activation
Must be run every time you open a new terminal.
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Labeling & Data Prep
Always follow this sequence when adding new data:

1. **Remove Duplicates:** `python scripts/deduplicate.py`
   - Cleans up identical images before processing.
2. **Interactive Labeling:** `python scripts/label_images.py`
   - Renames and organizes your raw images from `blood-images/` into `dataset/synthetic_train/`.
3. **Generate Masks:** `python scripts/generate_masks.py`
   - Automatically creates black-and-white segmentation masks for your training data.
4. **Build Master Labels:** `python scripts/build_labels.py`
   - Compiles the final `synthetic_labels.json` file from your renamed filenames and extracts important metadata.
5. **Audit Data:** `python scripts/audit_images.py`
   - Checks if all images are correctly labeled and verifies the dataset distributions.

## 🧠 Training & Evaluation

### 1. Train Segmentation Model
- **Command:** `python scripts/train_seg.py`
  - Goal: Teaches the AI to find the blood stain in the image.
  - Output: `models/seg_best.pt`

### 2. Train Regression Model
- **Command:** `python scripts/train_reg.py`
  - Goal: Teaches the AI to estimate volume (mL) from the stain.
  - Output: `models/reg_best.pt`

### 3. Evaluate Results
- **Command:** `python scripts/evaluate.py`
  - Tests the models on your real clinical images and calculates the Error (MAE).

## 📱 Mobile Conversion (TFLite)

### Step 1: Export to ONNX
- **Command:** `python scripts/export_onnx.py`
  - Output: `models/seg_model.onnx` & `models/reg_model.onnx`

### Step 2: Convert to TFLite (NEW!)
- **Install dependencies:** `pip install -r requirements_tflite.txt`
- **Convert:** `python scripts/convert_to_tflite.py`
  - Output: `models/seg_model.tflite` & `models/reg_model.tflite`
  - See `TFLITE_CONVERSION_GUIDE.md` for detailed instructions

### Step 3: Deploy to React Native Expo
- **Copy models:** `cp models/*.tflite <your-expo-project>/assets/models/`
- See `REACT_NATIVE_INTEGRATION.md` for complete integration guide

## 🌐 MVP Prototype
- **Run Web App:** `python mvp_app/app.py`
  - Open: `http://localhost:5050`

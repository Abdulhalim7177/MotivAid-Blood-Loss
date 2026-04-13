# MotivAid Blood Loss — Terminal Commands Reference

This document lists all the commands needed to run, train, and evaluate the blood loss AI models.

## 🚀 Getting Started

### 1. Environment Activation
Must be run every time you open a new terminal.
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Labeling & Data Prep
- **Interactive Labeling:** `python scripts/label_images.py`
  - Renames and organizes your raw images from `blood-images/` into `dataset/synthetic_train/`.
- **Audit Data:** `python scripts/audit_images.py`
  - Checks if all images in `dataset/real_test/` are correctly labeled in `labels.json`.
- **Generate Masks:** `python scripts/generate_masks.py`
  - Automatically creates black-and-white segmentation masks for your training data.
- **Build Master Labels:** `python scripts/build_labels.py`
  - Compiles the final `synthetic_labels.json` file from your renamed filenames.

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
- **Export to ONNX:** `python scripts/export_onnx.py`
- **Convert to TFLite:** (Usually done in Colab)
  - Output: `models/blood_loss_seg.tflite` & `models/blood_loss_reg.tflite`

## 🌐 MVP Prototype
- **Run Web App:** `cd mvp_app && python app.py`
  - Open: `http://localhost:5050`

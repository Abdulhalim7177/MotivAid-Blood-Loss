# 🚀 Quick Start: TFLite Conversion for React Native Expo

## ⏱️ 15-Minute Setup Guide

### Prerequisites Check
- [ ] Python 3.8+ installed
- [ ] Models exist: `models/seg_model.onnx` and `models/reg_model.onnx`
- [ ] React Native Expo project ready (or will create one)

---

## Part 1: Convert Models (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements_tflite.txt
```

**Expected output:**
```
Successfully installed onnx-1.14.0 onnx-tf-1.10.0 tensorflow-2.13.0 ...
```

### Step 2: Run Conversion
```bash
python scripts/convert_to_tflite.py
```

**Expected output:**
```
============================================================
  MotivAid — ONNX to TFLite Conversion
============================================================

============================================================
  Converting Segmentation Model
============================================================
  [1/4] Loading ONNX model...
  ✓ ONNX model loaded and validated
  [2/4] Converting ONNX to TensorFlow...
  [3/4] Exporting to SavedModel format...
  ✓ SavedModel exported to: models/seg_model_saved_model
  [4/4] Converting to TFLite...
  ✓ TFLite model saved: models/seg_model.tflite
  ✓ Model size: 7.23 MB

  Testing Segmentation Model...
  Input details:
    - image: shape=[1, 3, 256, 256], dtype=<class 'numpy.float32'>
  Output details:
    - mask: shape=[1, 1, 256, 256], dtype=<class 'numpy.float32'>
  Running inference with dummy input...
  ✓ Inference successful! Output shape: (1, 1, 256, 256)

============================================================
  Converting Regression Model
============================================================
  [1/4] Loading ONNX model...
  ✓ ONNX model loaded and validated
  [2/4] Converting ONNX to TensorFlow...
  [3/4] Exporting to SavedModel format...
  ✓ SavedModel exported to: models/reg_model_saved_model
  [4/4] Converting to TFLite...
  ✓ TFLite model saved: models/reg_model.tflite
  ✓ Model size: 2.14 MB

  Testing Regression Model...
  Input details:
    - image: shape=[1, 3, 224, 224], dtype=<class 'numpy.float32'>
    - surface_type: shape=[1, 16], dtype=<class 'numpy.float32'>
    - extra_features: shape=[1, 3], dtype=<class 'numpy.float32'>
  Output details:
    - log_ml: shape=[1, 1], dtype=<class 'numpy.float32'>
  Running inference with dummy input...
  ✓ Inference successful! Output shape: (1, 1)

============================================================
  Conversion Summary
============================================================
  Segmentation Model: ✓ Success
  Regression Model: ✓ Success

  🎉 All models converted successfully!

  Next steps for React Native Expo:
    1. Copy TFLite models to your Expo project:
       cp models/*.tflite <your-expo-project>/assets/models/
    2. Install TensorFlow.js for React Native:
       npx expo install @tensorflow/tfjs @tensorflow/tfjs-react-native
    3. Use the models in your app (see example code)

============================================================
```

### Step 3: Verify Files
```bash
ls -lh models/*.tflite
```

**Expected output:**
```
-rw-r--r-- 1 user user 7.2M May 17 10:30 models/seg_model.tflite
-rw-r--r-- 1 user user 2.1M May 17 10:30 models/reg_model.tflite
```

✅ **Part 1 Complete!** You now have TFLite models ready for mobile deployment.

---

## Part 2: Set Up React Native Expo (5 minutes)

### Option A: Create New Expo Project

```bash
# Create new Expo app
npx create-expo-app MotivAidApp
cd MotivAidApp

# Install dependencies
npx expo install @tensorflow/tfjs @tensorflow/tfjs-react-native expo-gl
npx expo install expo-camera expo-image-picker expo-file-system

# Create directory structure
mkdir -p assets/models
mkdir -p src/services
mkdir -p src/screens
```

### Option B: Add to Existing Expo Project

```bash
cd <your-expo-project>

# Install dependencies
npx expo install @tensorflow/tfjs @tensorflow/tfjs-react-native expo-gl
npx expo install expo-camera expo-image-picker expo-file-system

# Create directories if needed
mkdir -p assets/models
mkdir -p src/services
mkdir -p src/screens
```

### Copy Models

```bash
# From your MotivAid project root
cp models/*.tflite <path-to-expo-project>/assets/models/
```

### Update app.json

Add to your `app.json`:
```json
{
  "expo": {
    "assetBundlePatterns": [
      "**/*",
      "assets/models/*.tflite"
    ]
  }
}
```

✅ **Part 2 Complete!** Your Expo project is configured.

---

## Part 3: Implement Blood Loss Estimator (5 minutes)

### Step 1: Create Service File

Copy the complete code from `REACT_NATIVE_INTEGRATION.md` section "BloodLossEstimator Service" to:
```
src/services/BloodLossEstimator.ts
```

**Quick copy command:**
```bash
# This creates a minimal version - see full version in REACT_NATIVE_INTEGRATION.md
cat > src/services/BloodLossEstimator.ts << 'EOF'
// See REACT_NATIVE_INTEGRATION.md for complete implementation
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

export class BloodLossEstimator {
  private segModel: tf.GraphModel | null = null;
  private regModel: tf.GraphModel | null = null;

  async initialize(): Promise<void> {
    await tf.ready();
    // Load models here
  }

  async estimate(imageUri: string, surfaceType: string) {
    // Implement estimation logic
  }
}

export const bloodLossEstimator = new BloodLossEstimator();
EOF
```

### Step 2: Create Camera Screen

Copy the complete code from `REACT_NATIVE_INTEGRATION.md` section "Camera Screen Component" to:
```
src/screens/CameraScreen.tsx
```

### Step 3: Update App.tsx

```typescript
import React from 'react';
import CameraScreen from './src/screens/CameraScreen';

export default function App() {
  return <CameraScreen />;
}
```

✅ **Part 3 Complete!** Your app is ready to test.

---

## Part 4: Test Your App

### Start Development Server

```bash
npx expo start
```

### Test on Device

1. **iOS**: Scan QR code with Camera app
2. **Android**: Scan QR code with Expo Go app
3. **Simulator**: Press `i` (iOS) or `a` (Android)

### Expected Behavior

1. App loads and shows "Loading AI models..."
2. Models load successfully
3. Camera/photo picker buttons appear
4. Take/select a photo
5. Choose surface type
6. Tap "Estimate Blood Loss"
7. See results: volume (mL) and confidence (%)

---

## 🎯 Success Checklist

### Conversion
- [ ] Installed Python dependencies
- [ ] Ran conversion script successfully
- [ ] Verified `.tflite` files exist
- [ ] Models tested automatically

### Expo Setup
- [ ] Created/updated Expo project
- [ ] Installed npm dependencies
- [ ] Copied TFLite models to assets
- [ ] Updated `app.json`

### Implementation
- [ ] Created `BloodLossEstimator.ts`
- [ ] Created `CameraScreen.tsx`
- [ ] Updated `App.tsx`
- [ ] Started development server

### Testing
- [ ] App launches without errors
- [ ] Models load successfully
- [ ] Can take/select photos
- [ ] Estimation works
- [ ] Results display correctly

---

## 🐛 Quick Troubleshooting

### Problem: Conversion fails with "No module named 'onnx_tf'"
**Solution:**
```bash
pip uninstall onnx-tf
pip install onnx-tensorflow
```

### Problem: "ONNX model not found"
**Solution:**
```bash
# Export ONNX models first
python scripts/export_onnx.py
# Then convert
python scripts/convert_to_tflite.py
```

### Problem: Expo app crashes on launch
**Solution:**
```bash
# Clear cache and restart
npx expo start -c
```

### Problem: "Cannot find module '@tensorflow/tfjs'"
**Solution:**
```bash
# Reinstall dependencies
npx expo install @tensorflow/tfjs @tensorflow/tfjs-react-native expo-gl
```

### Problem: Models not loading in app
**Solution:**
1. Check `app.json` has `assetBundlePatterns`
2. Verify models are in `assets/models/`
3. Restart Expo server: `npx expo start -c`

### Problem: Slow inference
**Solution:**
1. Test on real device (not simulator)
2. Use production build (not development)
3. Consider native TFLite runtime

---

## 📚 Next Steps

### Immediate
- [ ] Test with real blood stain images
- [ ] Verify accuracy of estimates
- [ ] Test on multiple devices

### Short-term
- [ ] Add result history
- [ ] Implement data export
- [ ] Add offline support
- [ ] Improve UI/UX

### Long-term
- [ ] Optimize performance (INT8 quantization)
- [ ] Add backend sync
- [ ] Implement user authentication
- [ ] Add analytics

---

## 📖 Documentation Reference

| Document | When to Use |
|----------|-------------|
| `TFLITE_SUMMARY.md` | Overview and quick reference |
| `TFLITE_CONVERSION_GUIDE.md` | Detailed conversion instructions |
| `REACT_NATIVE_INTEGRATION.md` | Complete React Native code |
| `ARCHITECTURE.md` | Understand system design |
| `COMMANDS.md` | Quick command reference |

---

## 🎉 You're Done!

If you've completed all steps, you now have:
- ✅ TFLite models optimized for mobile
- ✅ React Native Expo app with AI integration
- ✅ Working blood loss estimation on mobile devices

**Time to celebrate!** 🎊

Test your app with real images and see the AI in action!

---

## 💡 Pro Tips

1. **Start Simple**: Test with the example code first
2. **Test Early**: Run on real devices ASAP
3. **Monitor Performance**: Use React Native DevTools
4. **Iterate**: Start with FP16, optimize later if needed
5. **Document**: Keep notes on accuracy and performance

---

## 🆘 Need Help?

1. Check troubleshooting section above
2. Review detailed guides in documentation
3. Test models individually (seg first, then reg)
4. Verify preprocessing matches Python implementation
5. Check TensorFlow.js and Expo documentation

---

**Ready? Let's go!** 🚀

```bash
python scripts/convert_to_tflite.py
```

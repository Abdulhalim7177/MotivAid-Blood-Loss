# TFLite Conversion Guide for MotivAid Blood Loss Models

## Overview

This guide walks you through converting your PyTorch models to TensorFlow Lite format for deployment in React Native Expo applications.

## Current Model Status

### Models Available
- ✅ **Segmentation Model** (`models/seg_best.pt`) - PyTorch checkpoint
- ✅ **Regression Model** (`models/reg_best.pt`) - PyTorch checkpoint
- ✅ **ONNX Exports** (`models/seg_model.onnx`, `models/reg_model.onnx`) - Intermediate format

### Target Format
- 🎯 **TFLite Models** - For React Native Expo deployment

## Conversion Pipeline

```
PyTorch (.pt) → ONNX (.onnx) → TensorFlow SavedModel → TFLite (.tflite)
     ✅              ✅                  ⏳                    ⏳
```

## Step-by-Step Conversion

### Step 1: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements_tflite.txt

# Or install individually
pip install onnx onnx-tf tensorflow numpy pillow
```

**Note**: If you encounter issues with `onnx-tf`, try:
```bash
pip install onnx-tensorflow
```

### Step 2: Verify ONNX Models Exist

```bash
# Check if ONNX models are present
ls -lh models/*.onnx

# If not present, export from PyTorch first
python scripts/export_onnx.py
```

Expected output:
```
models/seg_model.onnx      (~9-15 MB)
models/reg_model.onnx      (~2-5 MB)
```

### Step 3: Convert to TFLite

```bash
# Run the conversion script
python scripts/convert_to_tflite.py
```

This will:
1. Load ONNX models
2. Convert to TensorFlow SavedModel format
3. Optimize and convert to TFLite
4. Test the converted models
5. Save TFLite files to `models/` directory

Expected output:
```
models/seg_model.tflite    (~4-8 MB with FP16 quantization)
models/reg_model.tflite    (~1-3 MB with FP16 quantization)
```

### Step 4: Verify Conversion

The script automatically tests the models, but you can manually verify:

```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='models/seg_model.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# Test inference
dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print("Inference successful! Output shape:", output.shape)
```

## Model Specifications

### Segmentation Model (seg_model.tflite)

**Architecture**: U-Net with MobileNetV2 backbone

**Input**:
- Name: `image`
- Shape: `[1, 3, 256, 256]`
- Type: `float32`
- Format: RGB image, normalized with ImageNet stats

**Output**:
- Name: `mask`
- Shape: `[1, 1, 256, 256]`
- Type: `float32`
- Format: Binary mask (0-1), threshold at 0.5

**Preprocessing**:
```python
# Resize to 256x256
# Normalize: (pixel / 255.0 - mean) / std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### Regression Model (reg_model.tflite)

**Architecture**: MobileNetV3-Small with custom head

**Inputs**:
1. `image`: `[1, 3, 224, 224]` - Masked RGB image
2. `surface_type`: `[1, 16]` - One-hot encoded surface type
3. `extra_features`: `[1, 3]` - [distance/100, lighting, clot]

**Output**:
- Name: `log_ml`
- Shape: `[1, 1]`
- Type: `float32`
- Format: Natural log of volume in mL (apply `exp()` to get mL)

**Surface Type Encoding**:
```python
SURFACE_MAP = {
    'bowl': 0, 'container': 1, 'pad': 2, 'pampers': 3,
    'drape': 4, 'floor': 5, 'cloth': 6, 'bedsheet': 7,
    'towel': 8, 'gauze': 9, 'sheet': 10,
    'floor-and-cloth': 11, 'cloth-and-floor': 12,
    'pad-and-container': 13, 'pad-and-floor': 14,
    'other': 15
}
```

**Extra Features**:
- `distance_cm / 100.0` - Camera distance normalized
- `lighting` - 0.0 (daylight), 1.0 (LED), 2.0 (dim)
- `has_clot` - 0.0 (no), 1.0 (yes)

## Optimization Options

### 1. FP16 Quantization (Default)
- Reduces model size by ~50%
- Minimal accuracy loss
- Good balance for mobile deployment

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

### 2. INT8 Quantization (Smaller, requires calibration)
- Reduces model size by ~75%
- Requires representative dataset for calibration
- May have slight accuracy loss

```python
def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 3, 256, 256).astype(np.float32)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
```

### 3. Dynamic Range Quantization
- Fastest conversion
- Moderate size reduction

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

## Troubleshooting

### Issue: "No module named 'onnx_tf'" or "cannot import name 'mapping' from 'onnx'"

**Solution**: Use `onnx2tf` instead of the older `onnx-tf`:
```bash
pip uninstall onnx-tf
pip install onnx2tf
```

The conversion script has been updated to use `onnx2tf`, which is more actively maintained and compatible with newer ONNX versions.

### Issue: "ONNX model validation failed"

**Solution**: Re-export ONNX models with correct opset version:
```bash
python scripts/export_onnx.py
```

### Issue: "TensorFlow conversion error"

**Possible causes**:
1. Incompatible ONNX opset version
2. Unsupported operations in model

**Solution**: Try using `onnx2tf` instead:
```bash
pip install onnx2tf
onnx2tf -i models/seg_model.onnx -o models/seg_tf/
```

Then convert SavedModel to TFLite:
```python
converter = tf.lite.TFLiteConverter.from_saved_model('models/seg_tf/')
tflite_model = converter.convert()
with open('models/seg_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Issue: "Model size too large"

**Solution**: Apply INT8 quantization or use model pruning:
```bash
# Use INT8 quantization
python scripts/convert_to_tflite.py --quantize int8

# Or manually prune model before export
```

### Issue: "Inference results differ from PyTorch"

**Causes**:
1. Different preprocessing
2. Quantization effects
3. Numerical precision differences

**Solution**: 
- Verify preprocessing matches exactly
- Use FP32 instead of FP16 for testing
- Compare intermediate layer outputs

## Performance Benchmarks

### Model Sizes

| Model | PyTorch | ONNX | TFLite (FP32) | TFLite (FP16) | TFLite (INT8) |
|-------|---------|------|---------------|---------------|---------------|
| Segmentation | ~15 MB | ~14 MB | ~14 MB | ~7 MB | ~4 MB |
| Regression | ~5 MB | ~4 MB | ~4 MB | ~2 MB | ~1 MB |

### Inference Speed (Estimated)

| Device | Segmentation | Regression | Total |
|--------|--------------|------------|-------|
| iPhone 12 | ~50ms | ~20ms | ~70ms |
| Pixel 5 | ~80ms | ~30ms | ~110ms |
| Mid-range Android | ~150ms | ~50ms | ~200ms |

*Note: Actual performance depends on device, optimization, and runtime*

## Next Steps

1. ✅ Convert models to TFLite
2. 📱 Integrate into React Native Expo (see `REACT_NATIVE_INTEGRATION.md`)
3. 🧪 Test on real devices
4. 📊 Benchmark performance
5. 🎯 Optimize if needed

## Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [ONNX Documentation](https://onnx.ai/onnx/)
- [TensorFlow.js React Native](https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native)
- [Expo Documentation](https://docs.expo.dev/)

## Support

For issues specific to:
- **Model conversion**: Check TensorFlow and ONNX documentation
- **React Native integration**: See `REACT_NATIVE_INTEGRATION.md`
- **Model accuracy**: Verify preprocessing and quantization settings

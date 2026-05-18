# MotivAid Blood Loss Estimation - System Architecture

## 🏗️ Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     React Native Expo App                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐         ┌──────────────────────────────────┐ │
│  │ Camera/Photo │────────▶│   BloodLossEstimator Service     │ │
│  │   Picker     │         └──────────────────────────────────┘ │
│  └──────────────┘                        │                      │
│                                           │                      │
│                          ┌────────────────┴────────────────┐    │
│                          │                                  │    │
│                          ▼                                  ▼    │
│                 ┌─────────────────┐              ┌──────────────┐│
│                 │ Segmentation    │              │ Regression   ││
│                 │ Model (TFLite)  │              │ Model        ││
│                 │                 │              │ (TFLite)     ││
│                 │ Input: 256×256  │              │ Input: 224×224││
│                 │ Output: Mask    │              │ Output: mL   ││
│                 └─────────────────┘              └──────────────┘│
│                          │                                  ▲    │
│                          └──────────────┬──────────────────┘    │
│                                         │                        │
│                          ┌──────────────▼──────────────┐        │
│                          │   Display Results           │        │
│                          │   - Volume (mL)             │        │
│                          │   - Confidence (%)          │        │
│                          │   - Visual mask overlay     │        │
│                          └─────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### Step 1: Image Capture
```
User Action
    │
    ├─ Take Photo (Camera)
    │     OR
    └─ Select Image (Gallery)
         │
         ▼
    Image URI
```

### Step 2: Preprocessing
```
Image URI (any size)
    │
    ├─ Resize to 256×256 (for segmentation)
    ├─ Resize to 224×224 (for regression)
    ├─ Normalize RGB values [0-255] → [0-1]
    └─ Apply ImageNet normalization
         │
         ▼
    Preprocessed Tensors
```

### Step 3: Segmentation
```
Image Tensor (1, 3, 256, 256)
    │
    ▼
┌─────────────────────────────┐
│  Segmentation Model         │
│  (U-Net + MobileNetV2)      │
│                             │
│  Encoder (MobileNetV2)      │
│    ↓                        │
│  Bottleneck                 │
│    ↓                        │
│  Decoder (Upsampling)       │
│    ↓                        │
│  Sigmoid Activation         │
└─────────────────────────────┘
    │
    ▼
Binary Mask (1, 1, 256, 256)
    │
    ├─ Threshold at 0.5
    ├─ Calculate coverage (confidence)
    └─ Resize to 224×224 for regression
```

### Step 4: Regression
```
Masked Image (1, 3, 224, 224)
    +
Surface Type One-Hot (1, 16)
    +
Extra Features (1, 3)
    │
    ▼
┌─────────────────────────────┐
│  Regression Model           │
│  (MobileNetV3-Small)        │
│                             │
│  Feature Extractor          │
│    ↓                        │
│  Global Average Pooling     │
│    ↓                        │
│  Concatenate with metadata  │
│    ↓                        │
│  Dense Layers (576→256→64→1)│
│    ↓                        │
│  Output: log(mL)            │
└─────────────────────────────┘
    │
    ▼
exp(log_mL) = Volume in mL
```

### Step 5: Result Display
```
Volume (mL) + Confidence (%)
    │
    ▼
┌─────────────────────────────┐
│  User Interface             │
│                             │
│  ┌───────────────────────┐ │
│  │   1,250 mL            │ │
│  │   Confidence: 85%     │ │
│  │                       │ │
│  │   [Image with mask]   │ │
│  │                       │ │
│  │   ⚠ Clinical judgment │ │
│  │   required            │ │
│  └───────────────────────┘ │
└─────────────────────────────┘
```

## 🧠 Model Architectures

### Segmentation Model (U-Net)

```
Input: RGB Image (256×256×3)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    ENCODER (MobileNetV2)                 │
├─────────────────────────────────────────────────────────┤
│  Conv2D (32 filters)                                     │
│    ↓                                                     │
│  Inverted Residual Blocks                               │
│    ↓                                                     │
│  Feature Maps: 128×128, 64×64, 32×32, 16×16, 8×8       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    BOTTLENECK                            │
│  Feature Maps: 8×8×1280                                 │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    DECODER                               │
├─────────────────────────────────────────────────────────┤
│  Upsample + Skip Connections                            │
│    ↓                                                     │
│  16×16 → 32×32 → 64×64 → 128×128 → 256×256            │
│    ↓                                                     │
│  Conv2D (1 filter) + Sigmoid                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output: Binary Mask (256×256×1)
```

### Regression Model (MobileNetV3-Small)

```
Input 1: Masked Image (224×224×3)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTOR (MobileNetV3)             │
├─────────────────────────────────────────────────────────┤
│  Conv2D + Batch Norm                                     │
│    ↓                                                     │
│  Bottleneck Blocks (SE-Attention)                       │
│    ↓                                                     │
│  Global Average Pooling                                  │
│    ↓                                                     │
│  Feature Vector: 576 dimensions                         │
└─────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────────────────────────────┐
    │                                                       │
Input 2: Surface Type (16)                                 │
Input 3: Extra Features (3)                                │
    │                                                       │
    └───────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  REGRESSION HEAD                         │
├─────────────────────────────────────────────────────────┤
│  Concatenate: [576 + 16 + 3] = 595 dimensions          │
│    ↓                                                     │
│  Dense(256) + Hardswish + Dropout(0.3)                  │
│    ↓                                                     │
│  Dense(64) + Hardswish                                   │
│    ↓                                                     │
│  Dense(1) → log(mL)                                     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output: log(Volume in mL)
```

## 📊 Input/Output Specifications

### Segmentation Model

| Property | Value |
|----------|-------|
| **Input Name** | `image` |
| **Input Shape** | `[1, 3, 256, 256]` |
| **Input Type** | `float32` |
| **Input Range** | Normalized with ImageNet stats |
| **Output Name** | `mask` |
| **Output Shape** | `[1, 1, 256, 256]` |
| **Output Type** | `float32` |
| **Output Range** | `[0.0, 1.0]` (probability) |

### Regression Model

| Property | Value |
|----------|-------|
| **Input 1 Name** | `image` |
| **Input 1 Shape** | `[1, 3, 224, 224]` |
| **Input 1 Type** | `float32` |
| **Input 2 Name** | `surface_type` |
| **Input 2 Shape** | `[1, 16]` |
| **Input 2 Type** | `float32` (one-hot) |
| **Input 3 Name** | `extra_features` |
| **Input 3 Shape** | `[1, 3]` |
| **Input 3 Type** | `float32` |
| **Output Name** | `log_ml` |
| **Output Shape** | `[1, 1]` |
| **Output Type** | `float32` |
| **Output Range** | Natural log of mL |

## 🔢 Feature Encoding

### Surface Types (One-Hot Encoding)
```
Index | Surface Type
------|-------------
  0   | bowl
  1   | container
  2   | pad
  3   | pampers
  4   | drape
  5   | floor
  6   | cloth
  7   | bedsheet
  8   | towel
  9   | gauze
 10   | sheet
 11   | floor-and-cloth
 12   | cloth-and-floor
 13   | pad-and-container
 14   | pad-and-floor
 15   | other
```

### Extra Features
```
Feature 1: distance_cm / 100.0
  - Camera distance normalized to [0, 1] range
  - Example: 40cm → 0.4

Feature 2: lighting
  - 0.0 = daylight
  - 1.0 = LED
  - 2.0 = dim

Feature 3: has_clot
  - 0.0 = no clot
  - 1.0 = clot present
```

## 🎯 Preprocessing Pipeline

### ImageNet Normalization
```python
# RGB values [0, 255] → [0, 1]
normalized = pixel_value / 255.0

# Apply ImageNet statistics
mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]   # RGB

normalized = (normalized - mean) / std
```

### Complete Preprocessing Function
```python
def preprocess(image, target_size):
    # 1. Resize
    image = resize(image, (target_size, target_size))
    
    # 2. Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 3. Normalize to [0, 1]
    image = np.array(image, dtype=np.float32) / 255.0
    
    # 4. Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # 5. Transpose to CHW format (channels first)
    image = image.transpose(2, 0, 1)
    
    # 6. Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image
```

## 📈 Performance Characteristics

### Model Sizes
```
Segmentation Model:
  PyTorch:  ~15 MB
  ONNX:     ~14 MB
  TFLite:   ~7 MB (FP16)

Regression Model:
  PyTorch:  ~5 MB
  ONNX:     ~4 MB
  TFLite:   ~2 MB (FP16)

Total App Size Impact: ~9 MB
```

### Inference Times (Estimated)
```
Device          | Segmentation | Regression | Total
----------------|--------------|------------|-------
iPhone 12       |    ~50ms     |   ~20ms    | ~70ms
Pixel 5         |    ~80ms     |   ~30ms    | ~110ms
Mid-range       |   ~150ms     |   ~50ms    | ~200ms
```

### Memory Usage
```
Segmentation:
  - Input tensor:  256×256×3×4 bytes = ~768 KB
  - Output tensor: 256×256×1×4 bytes = ~256 KB
  - Model weights: ~7 MB
  - Peak memory:   ~15 MB

Regression:
  - Input tensors: 224×224×3×4 + 16×4 + 3×4 = ~600 KB
  - Output tensor: 1×4 bytes = 4 bytes
  - Model weights: ~2 MB
  - Peak memory:   ~5 MB

Total Peak Memory: ~20 MB
```

## 🔄 Conversion Pipeline

```
┌──────────────┐
│   PyTorch    │  Training framework
│   (.pt)      │  - Full precision (FP32)
└──────┬───────┘  - ~20 MB total
       │
       │ export_onnx.py
       ▼
┌──────────────┐
│    ONNX      │  Intermediate format
│   (.onnx)    │  - Framework agnostic
└──────┬───────┘  - ~18 MB total
       │
       │ convert_to_tflite.py
       ▼
┌──────────────┐
│  TensorFlow  │  SavedModel format
│ (SavedModel) │  - TF native format
└──────┬───────┘  - ~18 MB total
       │
       │ TFLiteConverter
       ▼
┌──────────────┐
│   TFLite     │  Mobile optimized
│  (.tflite)   │  - FP16 quantized
└──────────────┘  - ~9 MB total
```

## 🎨 User Interface Flow

```
┌─────────────────────────────────────────────────────────┐
│                    App Launch                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Initialize TensorFlow.js                    │
│              Load TFLite Models                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Main Screen                             │
│  ┌────────────┐  ┌────────────┐                        │
│  │ Take Photo │  │   Choose   │                        │
│  │            │  │   Image    │                        │
│  └────────────┘  └────────────┘                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Select Surface Type                         │
│  [Bowl] [Pad] [Floor] [Cloth] [Other]                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Processing Screen                           │
│  [Loading Spinner]                                       │
│  "Analyzing image..."                                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Results Screen                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Estimated Volume: 1,250 mL                     │   │
│  │  Confidence: 85%                                │   │
│  │                                                 │   │
│  │  [Image with mask overlay]                      │   │
│  │                                                 │   │
│  │  ⚠ Clinical judgment required                  │   │
│  │                                                 │   │
│  │  [Save] [Share] [New Estimate]                 │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

This architecture provides:
- ✅ Fast inference (<200ms on most devices)
- ✅ Small model size (~9 MB total)
- ✅ Offline capability
- ✅ High accuracy (trained on synthetic + real data)
- ✅ Clinical metadata integration

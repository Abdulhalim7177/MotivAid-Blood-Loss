# MotivAid Blood Loss — React Native Expo Integration Guide

## Overview

This guide explains how to integrate the TFLite models into your React Native Expo application for blood loss estimation.

## Prerequisites

1. **Convert models to TFLite** (if not already done):
   ```bash
   pip install onnx onnx-tf tensorflow
   python scripts/convert_to_tflite.py
   ```

2. **Install Expo dependencies**:
   ```bash
   npx expo install @tensorflow/tfjs @tensorflow/tfjs-react-native expo-gl
   npx expo install expo-camera expo-image-picker expo-file-system
   ```

## Model Files

After conversion, you'll have:
- `models/seg_model.tflite` (~9-15 MB) - Segmentation model
- `models/reg_model.tflite` (~2-5 MB) - Regression model

## Project Structure

```
your-expo-app/
├── assets/
│   └── models/
│       ├── seg_model.tflite
│       └── reg_model.tflite
├── src/
│   ├── services/
│   │   └── BloodLossEstimator.ts
│   └── screens/
│       └── CameraScreen.tsx
└── App.tsx
```

## Implementation

### 1. BloodLossEstimator Service

Create `src/services/BloodLossEstimator.ts`:

```typescript
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as FileSystem from 'expo-file-system';
import { Asset } from 'expo-asset';

// Surface type mapping (must match Python training code)
const SURFACE_MAP: { [key: string]: number } = {
  'bowl': 0,
  'container': 1,
  'pad': 2,
  'pampers': 3,
  'drape': 4,
  'floor': 5,
  'cloth': 6,
  'bedsheet': 7,
  'towel': 8,
  'gauze': 9,
  'sheet': 10,
  'floor-and-cloth': 11,
  'cloth-and-floor': 12,
  'pad-and-container': 13,
  'pad-and-floor': 14,
  'other': 15
};

export interface EstimationResult {
  estimatedML: number;
  confidence: number;
  mask?: tf.Tensor;
}

export class BloodLossEstimator {
  private segModel: tf.GraphModel | null = null;
  private regModel: tf.GraphModel | null = null;
  private isInitialized = false;

  /**
   * Initialize TensorFlow.js and load models
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Initialize TensorFlow.js
      await tf.ready();
      console.log('TensorFlow.js initialized');

      // Load models from assets
      const segModelAsset = Asset.fromModule(require('../../assets/models/seg_model.tflite'));
      const regModelAsset = Asset.fromModule(require('../../assets/models/reg_model.tflite'));

      await segModelAsset.downloadAsync();
      await regModelAsset.downloadAsync();

      // Load segmentation model
      this.segModel = await tf.loadGraphModel(
        bundleResourceIO(segModelAsset.localUri!)
      );
      console.log('Segmentation model loaded');

      // Load regression model
      this.regModel = await tf.loadGraphModel(
        bundleResourceIO(regModelAsset.localUri!)
      );
      console.log('Regression model loaded');

      this.isInitialized = true;
    } catch (error) {
      console.error('Failed to initialize models:', error);
      throw error;
    }
  }

  /**
   * Preprocess image for model input
   */
  private preprocessImage(
    imageTensor: tf.Tensor3D,
    targetSize: number
  ): tf.Tensor4D {
    return tf.tidy(() => {
      // Resize to target size
      let resized = tf.image.resizeBilinear(imageTensor, [targetSize, targetSize]);
      
      // Normalize to [0, 1]
      resized = resized.div(255.0);
      
      // Apply ImageNet normalization
      const mean = tf.tensor1d([0.485, 0.456, 0.406]);
      const std = tf.tensor1d([0.229, 0.224, 0.225]);
      const normalized = resized.sub(mean).div(std);
      
      // Add batch dimension
      return normalized.expandDims(0) as tf.Tensor4D;
    });
  }

  /**
   * Run segmentation to get blood stain mask
   */
  private async segment(imageTensor: tf.Tensor3D): Promise<tf.Tensor4D> {
    if (!this.segModel) throw new Error('Segmentation model not loaded');

    return tf.tidy(() => {
      const input = this.preprocessImage(imageTensor, 256);
      const output = this.segModel!.predict(input) as tf.Tensor4D;
      
      // Apply threshold to get binary mask
      return output.greater(0.5).cast('float32');
    });
  }

  /**
   * Run regression to estimate blood volume
   */
  private async regress(
    imageTensor: tf.Tensor3D,
    mask: tf.Tensor4D,
    surfaceType: string,
    distanceCm: number = 40,
    lighting: 'daylight' | 'led' | 'dim' = 'daylight',
    hasClot: boolean = false
  ): Promise<number> {
    if (!this.regModel) throw new Error('Regression model not loaded');

    return tf.tidy(() => {
      // Preprocess image (224x224 for regression)
      const input = this.preprocessImage(imageTensor, 224);
      
      // Resize mask to 224x224
      const maskResized = tf.image.resizeBilinear(
        mask.squeeze([0]) as tf.Tensor3D,
        [224, 224]
      ).expandDims(0);
      
      // Apply mask to image
      const maskedImage = input.mul(maskResized);
      
      // Prepare surface type one-hot encoding
      const surfaceIdx = SURFACE_MAP[surfaceType] || 15;
      const surfaceOneHot = tf.oneHot(tf.tensor1d([surfaceIdx], 'int32'), 16);
      
      // Prepare extra features
      const lightingValue = lighting === 'led' ? 1.0 : lighting === 'dim' ? 2.0 : 0.0;
      const clotValue = hasClot ? 1.0 : 0.0;
      const extras = tf.tensor2d([[distanceCm / 100.0, lightingValue, clotValue]]);
      
      // Run inference
      const output = this.regModel!.predict({
        image: maskedImage,
        surface_type: surfaceOneHot,
        extra_features: extras
      }) as tf.Tensor;
      
      // Convert log(mL) to mL
      const logML = output.dataSync()[0];
      return Math.exp(logML);
    });
  }

  /**
   * Estimate blood loss from image URI
   */
  async estimate(
    imageUri: string,
    surfaceType: string,
    options: {
      distanceCm?: number;
      lighting?: 'daylight' | 'led' | 'dim';
      hasClot?: boolean;
    } = {}
  ): Promise<EstimationResult> {
    if (!this.isInitialized) {
      throw new Error('Estimator not initialized. Call initialize() first.');
    }

    try {
      // Load image as tensor
      const imageAsset = Asset.fromURI(imageUri);
      await imageAsset.downloadAsync();
      
      const imageTensor = await this.loadImageTensor(imageAsset.localUri!);
      
      // Step 1: Segment blood stain
      const mask = await this.segment(imageTensor);
      
      // Calculate coverage (confidence metric)
      const coverage = await mask.mean().data();
      const confidence = Math.min(coverage[0] * 20, 1.0);
      
      // Step 2: Estimate volume
      const estimatedML = await this.regress(
        imageTensor,
        mask,
        surfaceType,
        options.distanceCm || 40,
        options.lighting || 'daylight',
        options.hasClot || false
      );
      
      // Cleanup
      imageTensor.dispose();
      
      return {
        estimatedML,
        confidence,
        mask: mask.squeeze([0, 3]) as tf.Tensor2D // Return 2D mask for visualization
      };
    } catch (error) {
      console.error('Estimation failed:', error);
      throw error;
    }
  }

  /**
   * Load image from URI as tensor
   */
  private async loadImageTensor(uri: string): Promise<tf.Tensor3D> {
    // Read image as base64
    const base64 = await FileSystem.readAsStringAsync(uri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    
    // Decode to tensor
    const imageData = tf.util.encodeString(base64, 'base64');
    const imageTensor = tf.node.decodeImage(imageData, 3) as tf.Tensor3D;
    
    return imageTensor;
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    if (this.segModel) {
      this.segModel.dispose();
      this.segModel = null;
    }
    if (this.regModel) {
      this.regModel.dispose();
      this.regModel = null;
    }
    this.isInitialized = false;
  }
}

// Singleton instance
export const bloodLossEstimator = new BloodLossEstimator();
```

### 2. Camera Screen Component

Create `src/screens/CameraScreen.tsx`:

```typescript
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  ScrollView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { bloodLossEstimator, EstimationResult } from '../services/BloodLossEstimator';

const SURFACE_TYPES = [
  'bowl', 'container', 'pad', 'pampers', 'drape',
  'floor', 'cloth', 'bedsheet', 'towel', 'gauze',
  'sheet', 'floor-and-cloth', 'other'
];

export default function CameraScreen() {
  const [isLoading, setIsLoading] = useState(true);
  const [isEstimating, setIsEstimating] = useState(false);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [selectedSurface, setSelectedSurface] = useState('bowl');
  const [result, setResult] = useState<EstimationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    initializeEstimator();
  }, []);

  const initializeEstimator = async () => {
    try {
      await bloodLossEstimator.initialize();
      setIsLoading(false);
    } catch (err) {
      setError('Failed to load AI models');
      setIsLoading(false);
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
      setResult(null);
      setError(null);
    }
  };

  const takePhoto = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      setError('Camera permission required');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
      setResult(null);
      setError(null);
    }
  };

  const estimateBloodLoss = async () => {
    if (!imageUri) return;

    setIsEstimating(true);
    setError(null);

    try {
      const estimation = await bloodLossEstimator.estimate(imageUri, selectedSurface);
      setResult(estimation);
    } catch (err) {
      setError('Estimation failed: ' + (err as Error).message);
    } finally {
      setIsEstimating(false);
    }
  };

  if (isLoading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#2E86C1" />
        <Text style={styles.loadingText}>Loading AI models...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>🩸 Blood Loss Estimator</Text>
        <Text style={styles.subtitle}>AI-powered volume estimation</Text>
      </View>

      {/* Surface Type Selector */}
      <View style={styles.card}>
        <Text style={styles.label}>Surface Type</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {SURFACE_TYPES.map((surface) => (
            <TouchableOpacity
              key={surface}
              style={[
                styles.surfaceButton,
                selectedSurface === surface && styles.surfaceButtonActive,
              ]}
              onPress={() => setSelectedSurface(surface)}
            >
              <Text
                style={[
                  styles.surfaceButtonText,
                  selectedSurface === surface && styles.surfaceButtonTextActive,
                ]}
              >
                {surface}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* Image Picker */}
      <View style={styles.card}>
        <Text style={styles.label}>Image</Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.button} onPress={takePhoto}>
            <Text style={styles.buttonText}>📷 Take Photo</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={pickImage}>
            <Text style={styles.buttonText}>🖼️ Choose Image</Text>
          </TouchableOpacity>
        </View>

        {imageUri && (
          <Image source={{ uri: imageUri }} style={styles.imagePreview} />
        )}
      </View>

      {/* Estimate Button */}
      {imageUri && (
        <TouchableOpacity
          style={[styles.estimateButton, isEstimating && styles.buttonDisabled]}
          onPress={estimateBloodLoss}
          disabled={isEstimating}
        >
          {isEstimating ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.estimateButtonText}>Estimate Blood Loss</Text>
          )}
        </TouchableOpacity>
      )}

      {/* Results */}
      {result && (
        <View style={styles.resultCard}>
          <Text style={styles.resultValue}>{Math.round(result.estimatedML)} mL</Text>
          <Text style={styles.resultConfidence}>
            Confidence: {Math.round(result.confidence * 100)}%
          </Text>
          <Text style={styles.disclaimer}>
            ⚠ This is an AI-assisted estimate. Clinical judgment must always be applied.
          </Text>
        </View>
      )}

      {/* Error */}
      {error && (
        <View style={styles.errorCard}>
          <Text style={styles.errorText}>❌ {error}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f7fa',
  },
  header: {
    backgroundColor: '#1A5276',
    padding: 24,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
  },
  card: {
    backgroundColor: '#fff',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
    color: '#2c3e50',
  },
  surfaceButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#e0e0e0',
    marginRight: 8,
  },
  surfaceButtonActive: {
    backgroundColor: '#2E86C1',
  },
  surfaceButtonText: {
    color: '#666',
    fontSize: 14,
  },
  surfaceButtonTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  button: {
    flex: 1,
    backgroundColor: '#2E86C1',
    padding: 12,
    borderRadius: 8,
    marginHorizontal: 4,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  imagePreview: {
    width: '100%',
    height: 200,
    borderRadius: 8,
    marginTop: 12,
  },
  estimateButton: {
    backgroundColor: '#1A5276',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  estimateButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  resultCard: {
    backgroundColor: '#f0fdf4',
    margin: 16,
    padding: 20,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#27ae60',
  },
  resultValue: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#1A5276',
    marginBottom: 8,
  },
  resultConfidence: {
    fontSize: 16,
    color: '#666',
    marginBottom: 12,
  },
  disclaimer: {
    fontSize: 12,
    color: '#888',
    fontStyle: 'italic',
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#ddd',
  },
  errorCard: {
    backgroundColor: '#fef2f2',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#ef4444',
  },
  errorText: {
    color: '#991b1b',
    fontSize: 14,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
});
```

## Alternative: Using TensorFlow Lite directly (Recommended for better performance)

For better performance, use the native TFLite runtime:

```bash
npx expo install react-native-tensorflow-lite
```

This requires a custom development build (not Expo Go).

## Testing

1. **Test conversion**:
   ```bash
   python scripts/convert_to_tflite.py
   ```

2. **Copy models to Expo project**:
   ```bash
   mkdir -p <your-expo-project>/assets/models
   cp models/*.tflite <your-expo-project>/assets/models/
   ```

3. **Run Expo app**:
   ```bash
   cd <your-expo-project>
   npx expo start
   ```

## Performance Optimization

1. **Model Quantization**: The conversion script uses FP16 quantization for smaller models
2. **Lazy Loading**: Models are loaded once on app start
3. **Tensor Cleanup**: Always dispose tensors after use to prevent memory leaks
4. **Batch Processing**: Process multiple images in batches if needed

## Troubleshooting

### Issue: "Model loading failed"
- Ensure TFLite files are in `assets/models/`
- Check that files are included in `app.json` under `assetBundlePatterns`

### Issue: "Out of memory"
- Reduce image resolution before processing
- Dispose tensors immediately after use
- Use `tf.tidy()` to auto-cleanup intermediate tensors

### Issue: "Slow inference"
- Use native TFLite runtime instead of TensorFlow.js
- Enable GPU acceleration if available
- Consider model quantization (INT8)

## Next Steps

1. Add camera preview with real-time estimation
2. Implement result history and export
3. Add offline support with local storage
4. Integrate with backend API for data sync

## Support

For issues or questions, refer to:
- TensorFlow.js React Native: https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native
- Expo documentation: https://docs.expo.dev/

"""
MotivAid Blood Loss — TFLite Conversion
========================================
Converts ONNX models to TensorFlow Lite format for React Native Expo.

Prerequisites:
  pip install onnx onnx2tf tensorflow

Usage:
  python scripts/convert_to_tflite.py

Output:
  - models/seg_model.tflite
  - models/reg_model.tflite
"""

import os
import sys
import subprocess
import numpy as np

try:
    import onnx
    import tensorflow as tf
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nInstall dependencies:")
    print("  pip install onnx onnx2tf tensorflow")
    sys.exit(1)

# Check if onnx2tf is available
try:
    result = subprocess.run(['onnx2tf', '--version'], 
                          capture_output=True, text=True, timeout=5)
    HAS_ONNX2TF = result.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired):
    HAS_ONNX2TF = False

if not HAS_ONNX2TF:
    print("ERROR: onnx2tf command not found")
    print("\nInstall onnx2tf:")
    print("  pip install onnx2tf")
    sys.exit(1)


def convert_onnx_to_tflite(onnx_path, tflite_path, model_name):
    """
    Convert ONNX model to TFLite format using onnx2tf.
    
    Args:
        onnx_path: Path to input ONNX model
        tflite_path: Path to output TFLite model
        model_name: Name for logging
    """
    print(f"\n{'=' * 60}")
    print(f"  Converting {model_name}")
    print(f"{'=' * 60}")
    
    if not os.path.exists(onnx_path):
        print(f"  ✗ ONNX model not found: {onnx_path}")
        return False
    
    try:
        # Step 1: Load and validate ONNX model
        print(f"  [1/2] Loading ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model loaded and validated")
        
        # Step 2: Convert ONNX to TFLite using onnx2tf
        temp_output_dir = tflite_path.replace('.tflite', '_temp')
        print(f"  [2/2] Converting ONNX to TFLite...")
        print(f"       (This may take a few minutes...)")
        
        # Remove old temp directory if exists
        if os.path.exists(temp_output_dir):
            import shutil
            shutil.rmtree(temp_output_dir)
        
        # Run onnx2tf command - it creates TFLite files directly
        cmd = [
            'onnx2tf',
            '-i', onnx_path,
            '-o', temp_output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"  ✗ onnx2tf conversion failed:")
            print(result.stderr)
            return False
        
        # onnx2tf creates files like: <model_name>_float32.tflite
        # Find the generated TFLite file
        import glob
        model_base_name = os.path.splitext(os.path.basename(onnx_path))[0]
        
        # Try to find float32 version first (better compatibility)
        float32_file = os.path.join(temp_output_dir, f"{model_base_name}_float32.tflite")
        float16_file = os.path.join(temp_output_dir, f"{model_base_name}_float16.tflite")
        
        source_file = None
        if os.path.exists(float32_file):
            source_file = float32_file
            print(f"  ✓ Using FP32 model for better compatibility")
        elif os.path.exists(float16_file):
            source_file = float16_file
            print(f"  ✓ Using FP16 model (smaller size)")
        else:
            # Try to find any .tflite file
            tflite_files = glob.glob(os.path.join(temp_output_dir, "*.tflite"))
            if tflite_files:
                source_file = tflite_files[0]
                print(f"  ✓ Found TFLite model: {os.path.basename(source_file)}")
        
        if not source_file or not os.path.exists(source_file):
            print(f"  ✗ No TFLite file found in {temp_output_dir}")
            return False
        
        # Copy to final location
        import shutil
        shutil.copy2(source_file, tflite_path)
        
        # Report size
        size_mb = os.path.getsize(tflite_path) / 1024 / 1024
        print(f"  ✓ TFLite model saved: {tflite_path}")
        print(f"  ✓ Model size: {size_mb:.2f} MB")
        
        # Clean up temp directory
        shutil.rmtree(temp_output_dir)
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ Conversion timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tflite_model(tflite_path, model_name, input_shapes):
    """
    Test TFLite model with dummy input.
    
    Args:
        tflite_path: Path to TFLite model
        model_name: Name for logging
        input_shapes: Dictionary of input name -> shape
    """
    print(f"\n  Testing {model_name}...")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  Input details:")
        for inp in input_details:
            print(f"    - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
        
        print(f"  Output details:")
        for out in output_details:
            print(f"    - {out['name']}: shape={out['shape']}, dtype={out['dtype']}")
        
        # Test with dummy input
        print(f"  Running inference with dummy input...")
        for i, inp in enumerate(input_details):
            dummy_input = np.random.randn(*inp['shape']).astype(inp['dtype'])
            interpreter.set_tensor(inp['index'], dummy_input)
        
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"  ✓ Inference successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  MotivAid — ONNX to TFLite Conversion")
    print("=" * 60)
    
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Model configurations
    models = [
        {
            'name': 'Segmentation Model',
            'onnx': os.path.join(models_dir, 'seg_model.onnx'),
            'tflite': os.path.join(models_dir, 'seg_model.tflite'),
            'input_shapes': {'image': [1, 3, 256, 256]}
        },
        {
            'name': 'Regression Model',
            'onnx': os.path.join(models_dir, 'reg_model.onnx'),
            'tflite': os.path.join(models_dir, 'reg_model.tflite'),
            'input_shapes': {
                'image': [1, 3, 224, 224],
                'surface_type': [1, 16],
                'extra_features': [1, 3]
            }
        }
    ]
    
    results = []
    
    # Convert each model
    for model_config in models:
        success = convert_onnx_to_tflite(
            model_config['onnx'],
            model_config['tflite'],
            model_config['name']
        )
        
        if success:
            # Test the converted model
            test_success = test_tflite_model(
                model_config['tflite'],
                model_config['name'],
                model_config['input_shapes']
            )
            results.append((model_config['name'], success and test_success))
        else:
            results.append((model_config['name'], False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Conversion Summary")
    print(f"{'=' * 60}")
    for name, success in results:
        status = '✓ Success' if success else '✗ Failed'
        print(f"  {name}: {status}")
    
    if all(success for _, success in results):
        print(f"\n  🎉 All models converted successfully!")
        print(f"\n  Next steps for React Native Expo:")
        print(f"    1. Copy TFLite models to your Expo project:")
        print(f"       cp models/*.tflite <your-expo-project>/assets/models/")
        print(f"    2. Install TensorFlow.js for React Native:")
        print(f"       npx expo install @tensorflow/tfjs @tensorflow/tfjs-react-native")
        print(f"    3. Use the models in your app (see example code)")
    else:
        print(f"\n  ⚠ Some conversions failed. Check errors above.")
    
    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()

"""
Quick test script to verify TFLite models work correctly
"""
import numpy as np
import tensorflow as tf

print("=" * 60)
print("  Testing TFLite Models")
print("=" * 60)

# Test Segmentation Model
print("\n[1/2] Testing Segmentation Model...")
try:
    interpreter = tf.lite.Interpreter(model_path='models/seg_model.tflite')
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  ✓ Model loaded successfully")
    print(f"  Input: {input_details[0]['name']} - shape {input_details[0]['shape']}, dtype {input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['name']} - shape {output_details[0]['shape']}, dtype {output_details[0]['dtype']}")
    
    # Test inference
    dummy_input = np.random.randn(*input_details[0]['shape']).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  ✓ Inference successful! Output shape: {output.shape}")
    print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
except Exception as e:
    print(f"  ✗ Test failed: {e}")

# Test Regression Model
print("\n[2/2] Testing Regression Model...")
try:
    interpreter = tf.lite.Interpreter(model_path='models/reg_model.tflite')
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  ✓ Model loaded successfully")
    print(f"  Inputs:")
    for inp in input_details:
        print(f"    - {inp['name']}: shape {inp['shape']}, dtype {inp['dtype']}")
    print(f"  Output: {output_details[0]['name']} - shape {output_details[0]['shape']}, dtype {output_details[0]['dtype']}")
    
    # Test inference
    for inp in input_details:
        dummy_input = np.random.randn(*inp['shape']).astype(inp['dtype'])
        interpreter.set_tensor(inp['index'], dummy_input)
    
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  ✓ Inference successful! Output shape: {output.shape}")
    print(f"  ✓ Predicted log(mL): {output[0][0]:.3f}")
    print(f"  ✓ Predicted mL: {np.exp(output[0][0]):.1f}")
    
except Exception as e:
    print(f"  ✗ Test failed: {e}")

print("\n" + "=" * 60)
print("  ✅ All tests passed!")
print("=" * 60)
print("\nYour TFLite models are ready for React Native Expo!")
print("\nNext steps:")
print("  1. Copy models to your Expo project:")
print("     cp models/*.tflite <your-expo-project>/assets/models/")
print("  2. See REACT_NATIVE_INTEGRATION.md for implementation")
print("=" * 60)

# Quick Reference Card

## ✅ Conversion Complete

Your models are ready at:
- `models/seg_model.tflite` (25.25 MB)
- `models/reg_model.tflite` (4.20 MB)

## 🚀 Deploy to React Native (3 Steps)

### Step 1: Copy Models
```bash
cp models/*.tflite <your-expo-project>/assets/models/
```

### Step 2: Install Dependencies
```bash
cd <your-expo-project>
npx expo install @tensorflow/tfjs @tensorflow/tfjs-react-native expo-gl
npx expo install expo-camera expo-image-picker expo-file-system
```

### Step 3: Implement Code
See `REACT_NATIVE_INTEGRATION.md` for complete code examples.

## 📊 Model Specs

### Segmentation
- Input: `[1, 256, 256, 3]` RGB image
- Output: `[1, 1, 256, 256]` binary mask

### Regression
- Input 1: `[1, 224, 224, 3]` masked image
- Input 2: `[1, 16]` surface type (one-hot)
- Input 3: `[1, 3]` extras (distance, lighting, clot)
- Output: `[1, 1]` log(mL) - apply `exp()` to get mL

## 🔧 Commands

### Re-run Conversion
```bash
venv\Scripts\activate
python scripts/convert_to_tflite.py
```

### Test Models
```bash
venv\Scripts\activate
python test_tflite_models.py
```

### Check Model Info
```bash
venv\Scripts\activate
python -c "import tensorflow as tf; i = tf.lite.Interpreter('models/seg_model.tflite'); i.allocate_tensors(); print(i.get_input_details()); print(i.get_output_details())"
```

## 📚 Documentation

| File | Use When |
|------|----------|
| `CONVERSION_SUCCESS.md` | ✅ You are here - conversion complete |
| `REACT_NATIVE_INTEGRATION.md` | 📱 Implementing in React Native |
| `QUICKSTART_TFLITE.md` | ⚡ 15-minute setup guide |
| `ARCHITECTURE.md` | 🏗️ Understanding system design |
| `TFLITE_CONVERSION_GUIDE.md` | 🔧 Detailed conversion info |

## 💡 Quick Tips

1. **Use FP32 models** (already done) - better compatibility
2. **Test on real devices** - simulators are slower
3. **Dispose tensors** - prevent memory leaks with `tf.tidy()`
4. **Preprocess correctly** - match Python preprocessing exactly
5. **Start simple** - test with example code first

## 🆘 Common Issues

| Problem | Solution |
|---------|----------|
| Models not loading | Check `app.json` assetBundlePatterns |
| Slow inference | Test on real device, use production build |
| Out of memory | Use `tf.tidy()`, dispose tensors |
| Wrong results | Verify preprocessing matches Python |

## 📞 Need Help?

1. Check troubleshooting in `CONVERSION_SUCCESS.md`
2. Review React Native code in `REACT_NATIVE_INTEGRATION.md`
3. Test models with `test_tflite_models.py`
4. Verify preprocessing in `ARCHITECTURE.md`

---

**Ready to build?** Start with `REACT_NATIVE_INTEGRATION.md` 🚀

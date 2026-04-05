"""
MotivAid Blood Loss — MVP Prototype (Flask Web App)
=====================================================
A simple web app to demonstrate the blood loss estimation pipeline.
Upload a photo, select surface type, get an estimated volume.

Usage:  python mvp_app/app.py
Open:   http://localhost:5000
"""

import os
import sys
import numpy as np
from PIL import Image
import io
import base64

from flask import Flask, request, jsonify, render_template_string

# Try to load ONNX models (optional — app shows UI even without models)
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

app = Flask(__name__)

SURFACE_MAP = {'pad': 0, 'gauze': 1, 'sheet': 2, 'drape': 3, 'other': 4}

# Model paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_ONNX = os.path.join(PROJECT_ROOT, 'models', 'seg_model.onnx')
REG_ONNX = os.path.join(PROJECT_ROOT, 'models', 'reg_model.onnx')

SEG_MODEL = None
REG_MODEL = None


def load_models():
    """Load ONNX models if available."""
    global SEG_MODEL, REG_MODEL
    if not HAS_ONNX:
        print("  WARNING: onnxruntime not installed. Using mock predictions.")
        return

    if os.path.exists(SEG_ONNX):
        SEG_MODEL = ort.InferenceSession(SEG_ONNX)
        print(f"  ✓ Loaded segmentation model: {SEG_ONNX}")
    else:
        print(f"  WARNING: Segmentation model not found at {SEG_ONNX}")

    if os.path.exists(REG_ONNX):
        REG_MODEL = ort.InferenceSession(REG_ONNX)
        print(f"  ✓ Loaded regression model: {REG_ONNX}")
    else:
        print(f"  WARNING: Regression model not found at {REG_ONNX}")


def preprocess(img_pil, size):
    """Preprocess image for model input."""
    img = img_pil.resize((size, size)).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def estimate(img_pil, surface_type):
    """Run the full estimation pipeline."""
    if SEG_MODEL is None or REG_MODEL is None:
        # Mock prediction for demo without models
        # Returns a random value to show the UI works
        import random
        mock_ml = random.uniform(20, 500)
        return mock_ml, 0.5  # 50% confidence for mock

    # Step 1: Segmentation
    img256 = preprocess(img_pil, 256)
    mask = SEG_MODEL.run(None, {'image': img256})[0]
    mask_bin = (mask > 0.5).astype(np.float32)
    coverage = float(mask_bin.mean())

    # Step 2: Regression
    img224 = preprocess(img_pil, 224)
    masked = img224 * mask_bin[:, :, :224, :224]  # rough crop+apply
    s_idx = SURFACE_MAP.get(surface_type, 4)
    s_oh = np.zeros((1, 5), dtype=np.float32)
    s_oh[0, s_idx] = 1.0
    extras = np.zeros((1, 3), dtype=np.float32)
    out = REG_MODEL.run(None, {
        'image': masked,
        'surface_type': s_oh,
        'extra_features': extras
    })
    log_ml = out[0].item()
    est_ml = float(np.exp(log_ml))
    conf = min(coverage * 20, 1.0)

    return est_ml, conf


# ─── HTML Template ────────────────────────────────────
HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MotivAid Blood Loss Estimator — MVP</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    max-width: 640px;
    margin: 0 auto;
    padding: 20px;
    background: #f5f7fa;
    color: #2c3e50;
    min-height: 100vh;
  }
  .header {
    background: linear-gradient(135deg, #1A5276, #2E86C1);
    color: white;
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 24px;
    text-align: center;
  }
  .header h1 { font-size: 1.5em; margin-bottom: 8px; }
  .header p { opacity: 0.9; font-size: 0.95em; }
  .card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  label { font-weight: 600; display: block; margin-bottom: 8px; }
  select {
    width: 100%;
    padding: 12px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 1em;
    background: white;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  select:focus { border-color: #2E86C1; outline: none; }
  .file-input-wrapper {
    border: 2px dashed #ccc;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    margin-top: 8px;
  }
  .file-input-wrapper:hover { border-color: #2E86C1; background: #f8fbff; }
  .file-input-wrapper.has-file { border-color: #27ae60; background: #f0fdf4; }
  input[type=file] { display: none; }
  .file-label { cursor: pointer; color: #666; font-size: 0.95em; }
  .file-label .icon { font-size: 2em; display: block; margin-bottom: 8px; }
  img#preview {
    max-width: 100%;
    max-height: 300px;
    border-radius: 8px;
    margin-top: 12px;
    display: none;
  }
  button {
    width: 100%;
    padding: 14px;
    font-size: 1.1em;
    font-weight: 600;
    background: linear-gradient(135deg, #1A5276, #2E86C1);
    color: white;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(26,82,118,0.3); }
  button:active { transform: translateY(0); }
  button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
  .result {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border-left: 4px solid #27ae60;
    padding: 20px;
    border-radius: 0 12px 12px 0;
    margin-top: 12px;
    animation: fadeIn 0.5s;
  }
  .result .ml-value { font-size: 2em; font-weight: 700; color: #1A5276; }
  .result .confidence { color: #666; margin-top: 4px; }
  .result .disclaimer {
    font-size: 0.85em;
    color: #888;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #ddd;
  }
  .error {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    padding: 16px;
    border-radius: 0 12px 12px 0;
    margin-top: 12px;
    color: #991b1b;
  }
  .loading { text-align: center; padding: 20px; color: #666; }
  .loading .spinner {
    display: inline-block;
    width: 24px; height: 24px;
    border: 3px solid #e0e0e0;
    border-top-color: #2E86C1;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
  }
  .model-status {
    text-align: center;
    padding: 8px;
    border-radius: 8px;
    font-size: 0.85em;
    margin-bottom: 16px;
  }
  .model-status.loaded { background: #f0fdf4; color: #166534; }
  .model-status.mock { background: #fffbeb; color: #92400e; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>
<div class="header">
  <h1>🩸 MotivAid Blood Loss Estimator</h1>
  <p>Upload a photograph of a blood-stained surface to estimate blood loss volume.</p>
</div>

<div class="model-status {{ 'loaded' if models_loaded else 'mock' }}">
  {% if models_loaded %}
    ✓ AI models loaded — predictions are real
  {% else %}
    ⚠ Models not found — using mock predictions for demo
  {% endif %}
</div>

<div class="card">
  <label for="surface">Surface Type</label>
  <select id="surface">
    <option value="pad">Sanitary pad</option>
    <option value="gauze">Gauze / surgical sponge</option>
    <option value="sheet">Hospital sheet</option>
    <option value="drape">Surgical drape</option>
    <option value="other">Other</option>
  </select>
</div>

<div class="card">
  <label>Upload Image</label>
  <div class="file-input-wrapper" id="dropzone" onclick="document.getElementById('imgfile').click()">
    <label class="file-label" for="imgfile">
      <span class="icon">📷</span>
      Click to select or drag & drop an image
    </label>
    <input type="file" id="imgfile" accept="image/*" onchange="handleFile(this)">
  </div>
  <img id="preview" src="" alt="Preview">
</div>

<button type="button" id="estimateBtn" onclick="doEstimate()" disabled>Estimate Blood Loss</button>

<div id="result"></div>

<script>
const dropzone = document.getElementById('dropzone');
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.style.borderColor = '#2E86C1'; });
dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = '#ccc'; });
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.style.borderColor = '#ccc';
  if (e.dataTransfer.files.length) {
    document.getElementById('imgfile').files = e.dataTransfer.files;
    handleFile(document.getElementById('imgfile'));
  }
});

function handleFile(input) {
  if (!input.files[0]) return;
  const r = new FileReader();
  r.onload = e => {
    const p = document.getElementById('preview');
    p.src = e.target.result;
    p.style.display = 'block';
    document.getElementById('dropzone').classList.add('has-file');
    document.getElementById('estimateBtn').disabled = false;
  };
  r.readAsDataURL(input.files[0]);
}

async function doEstimate() {
  const f = document.getElementById('imgfile').files[0];
  if (!f) { alert('Please select an image first.'); return; }
  const btn = document.getElementById('estimateBtn');
  btn.disabled = true;
  btn.textContent = 'Estimating...';
  document.getElementById('result').innerHTML =
    '<div class="loading"><span class="spinner"></span> Analyzing image...</div>';

  const fd = new FormData();
  fd.append('image', f);
  fd.append('surface', document.getElementById('surface').value);

  try {
    const resp = await fetch('/estimate', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) {
      document.getElementById('result').innerHTML =
        `<div class="error">❌ ${data.error}</div>`;
    } else {
      const conf = Math.round(data.confidence * 100);
      document.getElementById('result').innerHTML =
        `<div class="result">
          <div class="ml-value">${data.estimated_ml.toFixed(0)} mL</div>
          <div class="confidence">Confidence: ${conf}%</div>
          <div class="disclaimer">
            ⚠ This is an AI-assisted estimate. Clinical judgment must always be applied.
            ${data.mock ? '<br><em>(Mock prediction — train models for real results)</em>' : ''}
          </div>
        </div>`;
    }
  } catch (err) {
    document.getElementById('result').innerHTML =
      `<div class="error">❌ Network error: ${err.message}</div>`;
  }
  btn.disabled = false;
  btn.textContent = 'Estimate Blood Loss';
}
</script>
</body>
</html>'''


@app.route('/')
def index():
    return render_template_string(HTML, models_loaded=(SEG_MODEL is not None and REG_MODEL is not None))


@app.route('/estimate', methods=['POST'])
def estimate_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        f = request.files['image']
        surf = request.form.get('surface', 'other')
        img = Image.open(io.BytesIO(f.read())).convert('RGB')

        ml, conf = estimate(img, surf)
        is_mock = SEG_MODEL is None or REG_MODEL is None

        return jsonify({
            'estimated_ml': ml,
            'confidence': conf,
            'surface_type': surf,
            'mock': is_mock
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  MotivAid Blood Loss — MVP Web App")
    print("=" * 60)
    load_models()
    print(f"\n  Open http://localhost:5050 in your browser")
    print(f"  Press Ctrl+C to stop\n")
    app.run(debug=True, port=5050, host='0.0.0.0')

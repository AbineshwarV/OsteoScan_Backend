#!/usr/bin/env python3
"""
app_api.py

Flask wrapper that imports helpers from your existing app_gradcam.py
and exposes POST /predict for web frontends.

Place this file in the same folder as app_gradcam.py and run:
    python app_api.py
"""

import io
import base64
import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

import app_gradcam  # 👈 import module itself
from app_gradcam import (
    CLASS_NAMES,
    IMG_SIZE,
    HEATMAP_ALPHA,
    preprocess_pil,
    find_last_conv_layer,
    make_gradcam_heatmap,
    ensure_model_downloaded,
)

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def pil_to_dataurl(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def make_overlay_and_heatmap(original_pil: Image.Image, heatmap: np.ndarray, alpha=HEATMAP_ALPHA):
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original_pil.size, resample=Image.BILINEAR)
    heatmap_arr = np.asarray(heatmap_img).astype("float32") / 255.0

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap_arr)[:, :, :3]

    orig_arr = np.asarray(original_pil.convert("RGB")).astype("float32") / 255.0
    overlay = np.clip(orig_arr * (1 - alpha) + colored * alpha, 0, 1)
    overlay_pil = Image.fromarray(np.uint8(overlay * 255))
    heatmap_pil = Image.fromarray(np.uint8(colored * 255))

    return overlay_pil, heatmap_pil


app = Flask(__name__)
CORS(app)

print("[API] Ensuring model is available on disk...")
ensure_model_downloaded()
MODEL_PATH = app_gradcam.MODEL_PATH  # 👈 updated path from HF/local
print("[API] Loading model from:", MODEL_PATH)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("[API] Failed to load model:", e)
    raise

print("[API] Model loaded.")
try:
    last_conv_layer = find_last_conv_layer(model)
    print("[API] Last conv layer:", last_conv_layer)
except Exception as e:
    print("[API] Could not find a conv layer:", e)
    raise


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Use field name 'image'."}), 400

    file = request.files["image"]
    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {e}"}), 400

    try:
        x = preprocess_pil(pil_img, target_size=IMG_SIZE)
        preds = model.predict(x)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = np.asarray(preds)[0]
    except Exception as e:
        return jsonify({"error": f"Model predict failed: {e}"}), 500

    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
    pred_prob = float(preds[pred_idx])

    explain_index = request.form.get("explain_index")
    if explain_index is not None:
        try:
            explain_index = int(explain_index)
        except Exception:
            explain_index = pred_idx
    else:
        explain_index = pred_idx

    try:
        heatmap = make_gradcam_heatmap(x, model, last_conv_layer, explain_index)
    except Exception as e:
        return jsonify({"error": f"Grad-CAM error: {e}"}), 500

    overlay_pil, heatmap_pil = make_overlay_and_heatmap(pil_img, heatmap, alpha=HEATMAP_ALPHA)

    resp = {
        "label": pred_label,
        "index": pred_idx,
        "confidence": pred_prob,
        "original": pil_to_dataurl(pil_img),
        "heatmap": pil_to_dataurl(heatmap_pil),
        "overlay": pil_to_dataurl(overlay_pil),
    }
    return jsonify(resp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

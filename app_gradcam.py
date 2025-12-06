#!/usr/bin/env python3
"""
app_gradcam.py

Local predictor + Grad-CAM explanation (single-file).

Usage (local GUI):
    python app_gradcam.py

On server (Hugging Face):
    imported by app_api.py, no GUI used.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, Tk

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D

# ✅ NEW: for downloading model from Hugging Face Hub in Spaces
from huggingface_hub import snapshot_download

# -----------------------------
# USER SETTINGS (edit if needed)
# -----------------------------

# Base directory = folder where this file lives
BASE_DIR = Path(__file__).resolve().parent

# Original local Windows model folder (your existing path)
LOCAL_MODEL_DIR = Path(r"C:\Final_Year_Project\CustomCNN_3_knee_osteo_model")

# Default model folder inside the project (for server / HF Spaces)
DEFAULT_MODEL_DIR = BASE_DIR / "CustomCNN_3_knee_osteo_model"

# Hugging Face Hub model ID (you already created this)
HF_MODEL_ID = os.environ.get(
    "HF_MODEL_ID",
    "AbineshwarV/customcnn-3-knee-osteo-knee"  # fallback if env not set
)

# Choose which directory to use:
# - if your original Windows folder exists, use that (local dev)
# - otherwise, use the project-relative folder (server / HF Spaces)
if LOCAL_MODEL_DIR.exists():
    MODEL_DIR = LOCAL_MODEL_DIR
else:
    MODEL_DIR = DEFAULT_MODEL_DIR

# This is what load_model() will receive
MODEL_PATH = str(MODEL_DIR)

# Class labels in the SAME order used during training
CLASS_NAMES = ["Osteopenia", "Osteoporosis", "Normal"]

# Model input size
IMG_SIZE = (224, 224)

# Heatmap overlay alpha
HEATMAP_ALPHA = 0.4
# -----------------------------


def ensure_model_downloaded():
    """
    Ensure that CustomCNN_3_knee_osteo_model/ with model.weights.h5 exists.

    Behaviour:
    - If LOCAL_MODEL_DIR exists on disk, use that (your Windows path).
    - Else, expect the model folder inside the project (DEFAULT_MODEL_DIR).
      If not present, download from Hugging Face Hub (HF_MODEL_ID).
    """
    global MODEL_DIR, MODEL_PATH  # so that app_api sees the final path

    # ✅ 1) Local Windows dev: use your existing folder
    if LOCAL_MODEL_DIR.exists():
        MODEL_DIR = LOCAL_MODEL_DIR
        MODEL_PATH = str(MODEL_DIR)
        weights_file = MODEL_DIR / "model.weights.h5"
        if weights_file.exists():
            print("[MODEL] Using existing local model at:", MODEL_DIR)
            return
        else:
            print("[MODEL] Local folder exists but model.weights.h5 not found:", weights_file)

    # ✅ 2) Server / HF Spaces: project-relative folder
    MODEL_DIR = DEFAULT_MODEL_DIR
    MODEL_PATH = str(MODEL_DIR)
    weights_file = MODEL_DIR / "model.weights.h5"

    # If model already exists in repo folder, done
    if weights_file.exists():
        print("[MODEL] Using model folder in project dir:", MODEL_DIR)
        return

    # ✅ 3) Download from Hugging Face Hub if not present
    print("[MODEL] Model not found locally. Downloading from Hugging Face Hub:")
    print("        HF model id:", HF_MODEL_ID)

    # Download snapshot to a cache dir
    cache_dir = BASE_DIR / "hf_model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # This will download the repo locally
    repo_path = snapshot_download(
        repo_id=HF_MODEL_ID,
        cache_dir=str(cache_dir),
        local_files_only=False,
    )

    # We expect inside that repo: CustomCNN_3_knee_osteo_model/...
    repo_path = Path(repo_path)
    downloaded_model_dir = repo_path / "CustomCNN_3_knee_osteo_model"

    if not downloaded_model_dir.exists():
        raise RuntimeError(
            f"Downloaded HF repo does not contain 'CustomCNN_3_knee_osteo_model' folder.\n"
            f"Repo path: {repo_path}"
        )

    # Copy or point MODEL_DIR to the downloaded folder
    MODEL_DIR = downloaded_model_dir
    MODEL_PATH = str(MODEL_DIR)
    weights_file = MODEL_DIR / "model.weights.h5"

    if not weights_file.exists():
        raise RuntimeError(
            f"'model.weights.h5' not found in downloaded folder: {MODEL_DIR}"
        )

    print("[MODEL] Using model from Hugging Face snapshot at:", MODEL_DIR)


def preprocess_pil(pil_img: Image.Image, target_size=IMG_SIZE) -> np.ndarray:
    """Return a (1,H,W,3) float32 array scaled to [0,1]."""
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)


def find_last_conv_layer(model):
    """Find last convolutional layer name in a model (search from end)."""
    for layer in reversed(model.layers):
        # Works whether layer is instance of Conv2D or has "conv" in class name
        if isinstance(layer, Conv2D) or layer.__class__.__name__.lower().startswith("conv"):
            return layer.name
    raise ValueError("No Conv2D layer found in the model. Grad-CAM requires a convolutional layer.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.
    img_array: (1,H,W,3) preprocessed numpy array or tensor
    model: keras model
    last_conv_layer_name: name of the convolutional layer to use
    pred_index: class index to explain (if None, the model argmax is used)
    Returns: heatmap (H, W) normalized to [0,1] as numpy array
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        tape.watch(conv_outputs)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if isinstance(grads, (list, tuple)):
        grads = grads[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    pooled_grads = tf.cast(pooled_grads, conv_outputs.dtype)

    heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[[2], [0]])
    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        h = np.zeros((heatmap.shape[0], heatmap.shape[1]), dtype=np.float32)
        return h

    heatmap = heatmap / max_val
    return heatmap.numpy()


def save_and_show_gradcam(original_pil: Image.Image, heatmap, out_path: Path, label_text: str):
    """
    Create a heatmap overlay and show/save results.
    heatmap: numpy (h,w) in [0,1]
    """
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        original_pil.size, resample=Image.BILINEAR
    )
    heatmap_arr = np.asarray(heatmap_img).astype("float32") / 255.0

    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_arr)[:, :, :3]

    orig_arr = np.asarray(original_pil.convert("RGB")).astype("float32") / 255.0
    overlay = orig_arr * (1 - HEATMAP_ALPHA) + colored_heatmap * HEATMAP_ALPHA
    overlay = np.clip(overlay, 0, 1)

    overlay_pil = Image.fromarray(np.uint8(overlay * 255))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(original_pil)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(colored_heatmap)
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay_pil)
    axes[2].set_title(f"Overlay — {label_text}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    overlay_path = out_path.with_name(out_path.stem + "_gradcam.png")
    overlay_pil.save(overlay_path)
    print(f"[INFO] Grad-CAM overlay saved to: {overlay_path}")


def pick_image_file():
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
    )
    return path


def main():
    # Local GUI usage only (not used in HF Space)
    print("[INFO] Ensuring model...")
    ensure_model_downloaded()

    print("[INFO] Loading model:", MODEL_PATH)
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print("[ERROR] Failed to load model:", e)
        messagebox.showerror("Model load error", f"Failed to load model:\n{e}")
        return

    print("[INFO] Model loaded. Finding last conv layer...")
    try:
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"[INFO] Using last conv layer: {last_conv_layer_name}")
    except Exception as e:
        print("[ERROR] Could not find a conv layer:", e)
        messagebox.showerror("No conv layer", f"Could not find a Conv2D layer: {e}")
        return

    try:
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype="float32")
        model.predict(dummy)
    except Exception:
        pass

    while True:
        img_path = pick_image_file()
        if not img_path:
            print("[INFO] No image selected. Exiting.")
            break

        img_path = Path(img_path)
        print("[INFO] Selected:", img_path)

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("[ERROR] Could not open image:", e)
            messagebox.showerror("Open error", f"Failed to open image: {e}")
            continue

        x = preprocess_pil(pil_img, target_size=IMG_SIZE)
        preds = model.predict(x)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = np.asarray(preds)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
        pred_prob = float(preds[pred_idx])

        print(f"[RESULT] Predicted: {pred_label} (index {pred_idx}) — confidence {pred_prob:.4f}")

        try:
            heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name, pred_idx)
        except Exception as e:
            print("[ERROR] Could not compute Grad-CAM:", e)
            messagebox.showerror("Grad-CAM error", f"Could not compute Grad-CAM:\n{e}")
            continue

        save_and_show_gradcam(pil_img, heatmap, img_path, f"{pred_label} ({pred_prob:.2f})")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

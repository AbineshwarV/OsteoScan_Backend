#!/usr/bin/env python3
"""
app_gradcam.py

Local predictor + Grad-CAM explanation (single-file).

Usage:
    python app_gradcam.py

Pick an image from the file dialog. The script will predict and show Grad-CAM overlay.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ✅ Try tkinter, but don't crash if missing (Render)
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, Tk
    TK_AVAILABLE = True
except ImportError:
    tk = None
    filedialog = None
    messagebox = None
    Tk = None
    TK_AVAILABLE = False

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D

# ✅ Hugging Face Hub for downloading model
from huggingface_hub import snapshot_download

# -----------------------------
# USER SETTINGS (edit if needed)
# -----------------------------

# Base directory = folder where this file lives
BASE_DIR = Path(__file__).resolve().parent

# Name of the H5 model file
MODEL_FILENAME = "CustomCNN_3_knee_osteo_model.h5"

# Local Windows H5 path (after you saved it)
LOCAL_MODEL_FILE = Path(r"C:\Final_Year_Project") / MODEL_FILENAME

# Default H5 path in project (for Render)
DEFAULT_MODEL_FILE = BASE_DIR / MODEL_FILENAME

# Where to cache HF downloads
HF_CACHE_DIR = BASE_DIR / "hf_model_cache"

# Hugging Face model repo info
HF_REPO_ID = os.environ.get("HF_REPO_ID", "AbineshwarV/customcnn-3-knee-osteo-knee")
HF_REVISION = os.environ.get("HF_REVISION")  # optional

# This will be updated by ensure_model_downloaded()
MODEL_PATH = str(DEFAULT_MODEL_FILE)

# Class labels in the SAME order used during training
CLASS_NAMES = ["Osteopenia", "Osteoporosis", "Normal"]

# Model input size
IMG_SIZE = (224, 224)

# Heatmap overlay alpha
HEATMAP_ALPHA = 0.4
# -----------------------------


def ensure_model_downloaded():
    """
    Ensure a single H5 model file exists and set MODEL_PATH to it.

    Order:
    1. If LOCAL_MODEL_FILE exists (on your PC), use that.
    2. Else if DEFAULT_MODEL_FILE exists (in repo / after first download), use that.
    3. Else download from Hugging Face Hub into HF_CACHE_DIR and
       pick the .h5 file (prefer MODEL_FILENAME if present).
    """
    global MODEL_PATH

    # 1) Use local PC file
    if LOCAL_MODEL_FILE.exists():
        MODEL_PATH = str(LOCAL_MODEL_FILE)
        print("[MODEL] Using local H5 model:", MODEL_PATH)
        return

    # 2) Use project H5 (already present)
    if DEFAULT_MODEL_FILE.exists():
        MODEL_PATH = str(DEFAULT_MODEL_FILE)
        print("[MODEL] Using project H5 model:", MODEL_PATH)
        return

    # 3) Need to download from HF
    if not HF_REPO_ID:
        raise RuntimeError("HF_REPO_ID is not set and no local model file found.")

    print("[MODEL] H5 model not found locally. Downloading from Hugging Face Hub...")
    print("        repo_id =", HF_REPO_ID)
    if HF_REVISION:
        print("        revision =", HF_REVISION)

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download all repo files into cache dir
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        local_dir=str(HF_CACHE_DIR),
        local_dir_use_symlinks=False,
        repo_type="model",
    )

    local_dir = Path(local_dir)
    print("[MODEL] Files downloaded under:", local_dir)

    # Find .h5 file
    candidates = []
    for root, dirs, files in os.walk(local_dir):
        for name in files:
            if name.endswith(".h5"):
                candidates.append(Path(root) / name)

    if not candidates:
        raise RuntimeError(
            f"No .h5 files found in HF repo {HF_REPO_ID}. "
            "Upload your CustomCNN_3_knee_osteo_model.h5 there."
        )

    # Prefer file named exactly MODEL_FILENAME if present
    chosen = None
    for p in candidates:
        if p.name == MODEL_FILENAME:
            chosen = p
            break
    if chosen is None:
        chosen = candidates[0]

    print("[MODEL] Using downloaded H5 model:", chosen)

    # Also copy it into project root as DEFAULT_MODEL_FILE for next time (optional but nice)
    try:
        DEFAULT_MODEL_FILE.write_bytes(chosen.read_bytes())
        MODEL_PATH = str(DEFAULT_MODEL_FILE)
        print("[MODEL] Copied to project file:", DEFAULT_MODEL_FILE)
    except Exception as e:
        # If copy fails, just use the cache path
        print("[MODEL] Could not copy model to project dir:", e)
        MODEL_PATH = str(chosen)


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
        return np.zeros((heatmap.shape[0], heatmap.shape[1]), dtype=np.float32)
    heatmap = heatmap / max_val

    return heatmap.numpy()


def save_and_show_gradcam(original_pil: Image.Image, heatmap, out_path: Path, label_text: str):
    """Create a heatmap overlay and show/save results."""
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original_pil.size, resample=Image.BILINEAR)
    heatmap_arr = np.asarray(heatmap_img).astype("float32") / 255.0

    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_arr)[:, :, :3]

    orig_arr = np.asarray(original_pil.convert("RGB")).astype("float32") / 255.0
    overlay = np.clip(orig_arr * (1 - HEATMAP_ALPHA) + colored_heatmap * HEATMAP_ALPHA, 0, 1)

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
    if not TK_AVAILABLE:
        raise RuntimeError("Tkinter GUI is not available in this environment.")
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
    )
    return path


def main():
    # ✅ Ensure H5 model exists (local or from HF)
    ensure_model_downloaded()

    # Load model
    print("[INFO] Loading model from:", MODEL_PATH)
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print("[ERROR] Failed to load model:", e)
        if TK_AVAILABLE and messagebox is not None:
            messagebox.showerror("Model load error", f"Failed to load model:\n{e}")
        return

    print("[INFO] Model loaded. Finding last conv layer...")
    try:
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"[INFO] Using last conv layer: {last_conv_layer_name}")
    except Exception as e:
        print("[ERROR] Could not find a conv layer:", e)
        if TK_AVAILABLE and messagebox is not None:
            messagebox.showerror("No conv layer", f"Could not find a Conv2D layer: {e}")
        return

    # Warmup predict
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
            if TK_AVAILABLE and messagebox is not None:
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
            if TK_AVAILABLE and messagebox is not None:
                messagebox.showerror("Grad-CAM error", f"Could not compute Grad-CAM:\n{e}")
            continue

        save_and_show_gradcam(pil_img, heatmap, img_path, f"{pred_label} ({pred_prob:.2f})")

    print("[INFO] Done.")


if __name__ == "__main__":
    if TK_AVAILABLE:
        main()
    else:
        print("[INFO] Tkinter GUI is not available; cannot run local GUI mode here.")

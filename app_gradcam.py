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

# ✅ Try importing tkinter (GUI) – OK on your PC, missing on Render
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

# ✅ NEW: Hugging Face Hub for model download on Render
from huggingface_hub import snapshot_download

# -----------------------------
# USER SETTINGS (edit if needed)
# -----------------------------

# Base directory = folder where this file lives
BASE_DIR = Path(__file__).resolve().parent

# Original local Windows model folder (your existing path)
LOCAL_MODEL_DIR = Path(r"C:\Final_Year_Project\CustomCNN_3_knee_osteo_model")

# Default model folder inside the project (for Render)
DEFAULT_MODEL_DIR = BASE_DIR / "CustomCNN_3_knee_osteo_model"

# Choose which directory to use:
# - if your original Windows folder exists, use that (local dev)
# - otherwise, use the project-relative folder (Render / other machines)
if LOCAL_MODEL_DIR.exists():
    MODEL_DIR = LOCAL_MODEL_DIR
else:
    MODEL_DIR = DEFAULT_MODEL_DIR

# This is what load_model() will receive
MODEL_PATH = str(MODEL_DIR)

# Hugging Face repo ID (you can also set HF_REPO_ID env var if you want)
HF_REPO_ID = os.environ.get("HF_REPO_ID", "AbineshwarV/customcnn-3-knee-osteo-knee")

# Class labels in the SAME order used during training
CLASS_NAMES = ["Osteopenia", "Osteoporosis", "Normal"]

# Model input size
IMG_SIZE = (224, 224)

# Heatmap overlay alpha
HEATMAP_ALPHA = 0.4
# -----------------------------


# ✅ used by app_api.py before load_model(MODEL_PATH)
def ensure_model_downloaded():
    """
    Ensure that CustomCNN_3_knee_osteo_model/ with model.weights.h5 exists.

    Behaviour:
    - If LOCAL_MODEL_DIR exists on disk, use that (your Windows path).
    - Else, download from Hugging Face Hub, locate model.weights.h5
      recursively, and set MODEL_DIR to its parent folder.
    """
    global MODEL_DIR, MODEL_PATH  # so that app_api sees the final path

    # 1) Local Windows path (your dev machine) – if exists, use it
    if LOCAL_MODEL_DIR.exists():
        MODEL_DIR = LOCAL_MODEL_DIR
        MODEL_PATH = str(MODEL_DIR)
        weights_file = MODEL_DIR / "model.weights.h5"
        if weights_file.exists():
            print("[MODEL] Using existing local model at:", MODEL_DIR)
            return
        else:
            print("[MODEL] Local folder exists but model.weights.h5 not found:", weights_file)

    # 2) Project-relative folder (Render) – if already present, use it
    MODEL_DIR = DEFAULT_MODEL_DIR
    MODEL_PATH = str(MODEL_DIR)
    weights_file = MODEL_DIR / "model.weights.h5"
    if weights_file.exists():
        print("[MODEL] Using model folder in project dir:", MODEL_DIR)
        return

    # 3) Otherwise, download from Hugging Face Hub
    if not HF_REPO_ID:
        raise RuntimeError(
            "Model folder not found locally and HF_REPO_ID is not set.\n"
            "Set HF_REPO_ID to your Hugging Face repo id "
            "(e.g. 'AbineshwarV/customcnn-3-knee-osteo-knee')."
        )

    print("[MODEL] Model not found locally. Downloading from Hugging Face Hub...")
    print("        repo_id =", HF_REPO_ID)

    # Where to put downloaded files
    cache_dir = BASE_DIR / "hf_model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # snapshot_download returns the local path to the repo snapshot
    local_repo_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    local_repo_dir = Path(local_repo_dir)

    # Search recursively for model.weights.h5
    candidates = list(local_repo_dir.rglob("model.weights.h5"))
    if not candidates:
        raise RuntimeError(
            f"After download, model.weights.h5 not found under {local_repo_dir}.\n"
            "Make sure your HF repo contains config.json, metadata.json and model.weights.h5 "
            "together in some folder (for example, CustomCNN_3_knee_osteo_model/)."
        )

    # Take the first match (there should typically be only one)
    weights_file = candidates[0]
    MODEL_DIR = weights_file.parent
    MODEL_PATH = str(MODEL_DIR)

    print("[MODEL] Found model weights at:", weights_file)
    print("[MODEL] Using MODEL_DIR:", MODEL_DIR)


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
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original_pil.size, resample=Image.BILINEAR)
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
    if not TK_AVAILABLE:
        raise RuntimeError("Tkinter GUI is not available in this environment.")
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
    )
    return path


def main():
    print("[INFO] Loading model:", MODEL_PATH)
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

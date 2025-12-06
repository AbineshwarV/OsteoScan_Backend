#!/usr/bin/env python3
"""
app_gradcam.py

Local predictor + Grad-CAM explanation (single-file).

- Still works locally with your Windows folder.
- On servers (like Hugging Face), it downloads the model from:
  AbineshwarV/customcnn-3-knee-osteo-knee
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- tkinter is only needed for local GUI; wrap in try/except for servers ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, Tk
except ImportError:
    tk = None
    filedialog = None
    messagebox = None
    Tk = None

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D

# 🆕 Hugging Face Hub import
from huggingface_hub import snapshot_download

# -----------------------------
# USER SETTINGS (edit if needed)
# -----------------------------

# Base directory = folder where this file lives
BASE_DIR = Path(__file__).resolve().parent

# Original local Windows model folder (your existing path)
LOCAL_MODEL_DIR = Path(r"C:\Final_Year_Project\CustomCNN_3_knee_osteo_model")

# Default model folder inside the project (for servers)
DEFAULT_MODEL_DIR = BASE_DIR / "CustomCNN_3_knee_osteo_model"

# Hugging Face model repo (you already created this)
HF_REPO_ID = "AbineshwarV/customcnn-3-knee-osteo-knee"

# Decide initial MODEL_DIR
if LOCAL_MODEL_DIR.exists():
    MODEL_DIR = LOCAL_MODEL_DIR
else:
    MODEL_DIR = DEFAULT_MODEL_DIR

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
    - Else, check DEFAULT_MODEL_DIR inside the project.
    - If still not present, download from Hugging Face model repo:
        AbineshwarV/customcnn-3-knee-osteo-knee
    """
    global MODEL_DIR, MODEL_PATH

    # 1) Use original local folder if it exists
    if LOCAL_MODEL_DIR.exists():
        MODEL_DIR = LOCAL_MODEL_DIR
        MODEL_PATH = str(MODEL_DIR)
        weights_file = MODEL_DIR / "model.weights.h5"
        if weights_file.exists():
            print("[MODEL] Using existing local model at:", MODEL_DIR)
            return
        else:
            print("[MODEL] Local folder exists but model.weights.h5 not found:", weights_file)

    # 2) Use project-relative folder if it already exists
    MODEL_DIR = DEFAULT_MODEL_DIR
    MODEL_PATH = str(MODEL_DIR)
    weights_file = MODEL_DIR / "model.weights.h5"

    if weights_file.exists():
        print("[MODEL] Using model folder in project dir:", MODEL_DIR)
        return

    # 3) Download from Hugging Face model repo
    print("[MODEL] Model not found locally. Downloading from Hugging Face Hub:")
    print("        repo_id =", HF_REPO_ID)

    # This will download the repo to HF cache
    snapshot_dir = Path(
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
    )

    # Your repo structure:
    # AbineshwarV/customcnn-3-knee-osteo-knee
    # ├── .gitattributes
    # └── CustomCNN_3_knee_osteo_model/
    MODEL_DIR = snapshot_dir / "CustomCNN_3_knee_osteo_model"
    MODEL_PATH = str(MODEL_DIR)
    weights_file = MODEL_DIR / "model.weights.h5"

    if not weights_file.exists():
        raise RuntimeError(
            f"After downloading from Hugging Face, {weights_file} not found.\n"
            "Check that the repo contains 'CustomCNN_3_knee_osteo_model/model.weights.h5'."
        )

    print("[MODEL] Model ready at:", MODEL_DIR)


def preprocess_pil(pil_img: Image.Image, target_size=IMG_SIZE) -> np.ndarray:
    """Return a (1,H,W,3) float32 array scaled to [0,1]."""
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)


def find_last_conv_layer(model):
    """Find last convolutional layer name in a model (search from end)."""
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D) or layer.__class__.__name__.lower().startswith("conv"):
            return layer.name
    raise ValueError("No Conv2D layer found in the model. Grad-CAM requires a convolutional layer.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
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
    if Tk is None or filedialog is None:
        print("[WARN] Tkinter not available. Cannot open file dialog.")
        return None

    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
    )
    return path


def main():
    # 🆕 make sure model exists (local or download from HF)
    ensure_model_downloaded()

    print("[INFO] Loading model:", MODEL_PATH)
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print("[ERROR] Failed to load model:", e)
        if messagebox:
            messagebox.showerror("Model load error", f"Failed to load model:\n{e}")
        return

    print("[INFO] Model loaded. Finding last conv layer...")
    try:
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"[INFO] Using last conv layer: {last_conv_layer_name}")
    except Exception as e:
        print("[ERROR] Could not find a conv layer:", e)
        if messagebox:
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
            if messagebox:
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
            if messagebox:
                messagebox.showerror("Grad-CAM error", f"Could not compute Grad-CAM:\n{e}")
            continue

        save_and_show_gradcam(pil_img, heatmap, img_path, f"{pred_label} ({pred_prob:.2f})")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

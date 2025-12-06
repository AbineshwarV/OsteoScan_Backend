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
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers, models

# ✅ for downloading model from Hugging Face Hub
from huggingface_hub import snapshot_download

# -----------------------------
# USER SETTINGS (edit if needed)
# -----------------------------

# Base directory = folder where this file lives
BASE_DIR = Path(__file__).resolve().parent

# Original local Windows model folder (your existing path)
LOCAL_MODEL_DIR = Path(r"C:\Final_Year_Project\CustomCNN_3_knee_osteo_model")

# Default model folder inside the project (for non-Windows env)
DEFAULT_MODEL_DIR = BASE_DIR / "CustomCNN_3_knee_osteo_model"

# Class labels in the SAME order used during training
CLASS_NAMES = ["Osteopenia", "Osteoporosis", "Normal"]

# Model input size
IMG_SIZE = (224, 224)

# Heatmap overlay alpha
HEATMAP_ALPHA = 0.4

# Name of the weights file we expect (saved via model.save_weights)
WEIGHTS_FILENAME = "model.weights.h5"

# On Render, set this env var to your HF repo id, e.g. "AbineshwarV/customcnn-3-knee-osteo-knee"
HF_REPO_ID = os.environ.get("HF_REPO_ID")
HF_REVISION = os.environ.get("HF_REVISION")  # optional (branch/tag/commit), or None
# -----------------------------


def build_customcnn_model(input_shape=(224, 224, 3), num_classes=len(CLASS_NAMES)):
    """
    Recreate EXACTLY the same architecture you used when training.

    ⚠️ IMPORTANT:
    Replace the sample architecture below with your real training model
    (same layers, same order, same activation functions, same final units).
    """
    model = models.Sequential()

    # ---------- SAMPLE ARCHITECTURE (REPLACE WITH YOURS) ----------
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    # --------------------------------------------------------------

    return model


def ensure_model_downloaded():
    """
    Ensure that a folder with `model.weights.h5` exists somewhere we can use.

    Behaviour:
    - If LOCAL_MODEL_DIR exists on disk and has weights file, use that (your Windows path).
    - Else, use the project-relative folder DEFAULT_MODEL_DIR.
      If weights not present there either, try downloading from Hugging Face Hub using HF_REPO_ID.

    Returns:
        Path object pointing to the folder that contains model.weights.h5
    """
    # Prefer your original local folder (for development on your PC)
    local_weights = LOCAL_MODEL_DIR / WEIGHTS_FILENAME
    if LOCAL_MODEL_DIR.exists() and local_weights.exists():
        print("[MODEL] Using existing local model folder:", LOCAL_MODEL_DIR)
        return LOCAL_MODEL_DIR

    # Otherwise, use project-relative folder (works on Render too)
    model_dir = DEFAULT_MODEL_DIR
    weights_file = model_dir / WEIGHTS_FILENAME

    # If weights already exist there, we're done
    if weights_file.exists():
        print("[MODEL] Using model folder in project dir:", model_dir)
        return model_dir

    # If no HF repo configured, we cannot download automatically
    if not HF_REPO_ID:
        raise RuntimeError(
            f"Model weights not found at {local_weights} or {weights_file}, "
            "and HF_REPO_ID is not set.\n"
            "Set HF_REPO_ID, e.g. 'AbineshwarV/customcnn-3-knee-osteo-knee', "
            "and make sure it contains model.weights.h5 at the root."
        )

    print("[MODEL] Model not found locally. Downloading from Hugging Face Hub...")
    print("        repo_id =", HF_REPO_ID)
    if HF_REVISION:
        print("        revision =", HF_REVISION)

    # Download entire repo into DEFAULT_MODEL_DIR
    snapshot_download(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        repo_type="model",
    )

    # After download, check weights file
    if not weights_file.exists():
        raise RuntimeError(
            f"After download, {weights_file} not found.\n"
            "Make sure your HF repo contains model.weights.h5 at its root "
            "(or adjust WEIGHTS_FILENAME / path logic accordingly)."
        )

    print("[MODEL] Model downloaded to:", model_dir)
    return model_dir


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
    # Build a model mapping inputs -> (last_conv_output, predictions)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Forward pass
        conv_outputs, predictions = grad_model(img_tensor)

        # If model output is a list/tuple (multi-output), take the first tensor
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        # Determine target index
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        # Ensure we watch conv_outputs
        tape.watch(conv_outputs)

        # Score for target class
        class_channel = predictions[:, pred_index]

    # Gradients of the class output w.r.t. convolutional layer outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # If grads is list/tuple, take first
    if isinstance(grads, (list, tuple)):
        grads = grads[0]

    # Global average pooling of gradients over (H, W)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get conv outputs for the image: shape (H, W, C)
    conv_outputs = conv_outputs[0]

    # Convert pooled_grads dtype to match conv_outputs
    pooled_grads = tf.cast(pooled_grads, conv_outputs.dtype)

    # Compute weighted combination: tensordot over channels
    # conv_outputs: H x W x C, pooled_grads: C
    heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[[2], [0]])

    # Apply ReLU to keep only positive influences
    heatmap = tf.nn.relu(heatmap)

    # Normalize to [0,1]
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        # avoid div-by-zero
        h = np.zeros((heatmap.shape[0], heatmap.shape[1]), dtype=np.float32)
        return h
    heatmap = heatmap / max_val

    return heatmap.numpy()


def save_and_show_gradcam(original_pil: Image.Image, heatmap, out_path: Path, label_text: str):
    """
    Create a heatmap overlay and show/save results.
    heatmap: numpy (h,w) in [0,1]
    """
    # Resize heatmap to original image size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original_pil.size, resample=Image.BILINEAR)
    heatmap_arr = np.asarray(heatmap_img).astype("float32") / 255.0  # 0..1

    # Create color map (jet-like) using matplotlib
    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_arr)[:, :, :3]  # drop alpha channel

    # Convert original to float array [0,1]
    orig_arr = np.asarray(original_pil.convert("RGB")).astype("float32") / 255.0

    # Overlay
    overlay = orig_arr * (1 - HEATMAP_ALPHA) + colored_heatmap * HEATMAP_ALPHA
    overlay = np.clip(overlay, 0, 1)

    overlay_pil = Image.fromarray(np.uint8(overlay * 255))

    # Display side-by-side (Original | Heatmap | Overlay)
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

    # Save overlay next to original
    overlay_path = out_path.with_name(out_path.stem + "_gradcam.png")
    overlay_pil.save(overlay_path)
    print(f"[INFO] Grad-CAM overlay saved to: {overlay_path}")


def pick_image_file():
    if not TK_AVAILABLE:
        # On a headless server like Render, this won't be used anyway.
        raise RuntimeError("Tkinter GUI is not available in this environment.")
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
    )
    return path


def main():
    # Ensure we have weights somewhere
    model_dir = ensure_model_downloaded()
    weights_path = model_dir / WEIGHTS_FILENAME

    print("[INFO] Loading model weights from:", weights_path)
    try:
        model = build_customcnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(CLASS_NAMES))
        model.load_weights(str(weights_path))
    except Exception as e:
        print("[ERROR] Failed to load model weights:", e)
        if TK_AVAILABLE and messagebox is not None:
            messagebox.showerror("Model load error", f"Failed to load model weights:\n{e}")
        return

    # Find last conv layer
    print("[INFO] Model built. Finding last conv layer...")
    try:
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"[INFO] Using last conv layer: {last_conv_layer_name}")
    except Exception as e:
        print("[ERROR] Could not find a conv layer:", e)
        if TK_AVAILABLE and messagebox is not None:
            messagebox.showerror("No conv layer", f"Could not find a Conv2D layer: {e}")
        return

    # Warmup predict to initialize everything
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

        # model prediction
        preds = model.predict(x)

        # handle if model.predict returns list/tuple
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = np.asarray(preds)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
        pred_prob = float(preds[pred_idx])

        print(f"[RESULT] Predicted: {pred_label} (index {pred_idx}) — confidence {pred_prob:.4f}")

        # Grad-CAM heatmap
        try:
            heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name, pred_idx)
        except Exception as e:
            print("[ERROR] Could not compute Grad-CAM:", e)
            if TK_AVAILABLE and messagebox is not None:
                messagebox.showerror("Grad-CAM error", f"Could not compute Grad-CAM:\n{e}")
            continue

        # Save and show
        save_and_show_gradcam(pil_img, heatmap, img_path, f"{pred_label} ({pred_prob:.2f})")

    print("[INFO] Done.")


if __name__ == "__main__":
    if TK_AVAILABLE:
        main()
    else:
        print("[INFO] Tkinter GUI is not available; cannot run local GUI mode here.")

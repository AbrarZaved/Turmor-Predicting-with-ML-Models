"""
ML Model Module - Loads TensorFlow and Model ONCE at module level.

CRITICAL for Render Free Tier:
- TensorFlow is imported ONCE here
- Model is loaded ONCE when this module is first imported
- Thread limits are set BEFORE any TensorFlow operations
- NO TensorFlow imports anywhere else in the project
"""

import os
from pathlib import Path

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# =============================================================================
# HARD LIMITS FOR RENDER FREE TIER - MUST BE SET BEFORE MODEL LOAD
# =============================================================================
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Disable GPU (not available on Render free tier anyway)
tf.config.set_visible_devices([], "GPU")

# =============================================================================
# MODEL PATH CONFIGURATION
# =============================================================================
# Model is in the parent directory of brain_tumor_app
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR.parent / "1_brain_tumor_vgg16.keras"

# Alternative: If model is inside classifier/model/ directory
# MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.keras"

# =============================================================================
# LOAD MODEL ONCE AT MODULE IMPORT
# =============================================================================
print(f"[ml_model] Loading model from: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}\n"
        f"Please ensure '1_brain_tumor_vgg16.keras' is in the correct location."
    )

# Load model with compile=False (inference only, no training)
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

print(f"[ml_model] Model loaded successfully. Input shape: {model.input_shape}")

# =============================================================================
# CLASS NAMES (must match training order)
# =============================================================================
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

IMG_SIZE = 224

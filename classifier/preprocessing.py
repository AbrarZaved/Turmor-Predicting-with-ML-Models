"""
Preprocessing Module - Matches Training Pipeline.

This module replicates the EXACT preprocessing used during model training.
Based on the notebook:
- Image resize to 224x224
- VGG16 preprocess_input (scales to [-1, 1] range using ImageNet mean subtraction)

CRITICAL: Any deviation from training preprocessing will cause poor predictions!
"""

import numpy as np
from PIL import Image
from io import BytesIO
from django.conf import settings


def preprocess_image(image_file):
    """
    Preprocess an uploaded image for model inference.

    This function replicates the training preprocessing:
    1. Read image from file
    2. Convert to RGB (model expects 3 channels)
    3. Resize to 224x224 (matching training)
    4. Convert to numpy array
    5. Expand dimensions for batch (1, 224, 224, 3)
    6. Apply VGG16 preprocessing (mean subtraction, BGR conversion)

    Args:
        image_file: Django UploadedFile or file-like object

    Returns:
        numpy.ndarray: Preprocessed image with shape (1, 224, 224, 3)
    """
    # Import TensorFlow preprocessing here to match training exactly
    from tensorflow.keras.applications.vgg16 import preprocess_input

    # Read image from uploaded file
    # Using PIL for robust image handling across formats
    if hasattr(image_file, "read"):
        image_data = image_file.read()
        # Reset file pointer for potential reuse
        if hasattr(image_file, "seek"):
            image_file.seek(0)
        img = Image.open(BytesIO(image_data))
    else:
        img = Image.open(image_file)

    # Convert to RGB (handles grayscale, RGBA, etc.)
    # VGG16 expects 3 channels
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to model input size (224x224 as per training)
    img_size = settings.IMG_SIZE  # 224
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)

    # Convert to numpy array
    # Shape: (224, 224, 3) with values 0-255
    img_array = np.array(img, dtype=np.float32)

    # Expand dimensions for batch processing
    # Shape: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Apply VGG16 preprocessing
    # This performs:
    # 1. RGB to BGR conversion (VGG16 was trained on BGR)
    # 2. Mean subtraction using ImageNet means [103.939, 116.779, 123.68]
    img_array = preprocess_input(img_array)

    return img_array


def get_original_image_base64(image_file):
    """
    Convert uploaded image to base64 for display in browser.

    Args:
        image_file: Django UploadedFile

    Returns:
        str: Base64 encoded image string with data URI prefix
    """
    import base64

    # Read file content
    if hasattr(image_file, "read"):
        content = image_file.read()
        if hasattr(image_file, "seek"):
            image_file.seek(0)
    else:
        with open(image_file, "rb") as f:
            content = f.read()

    # Encode to base64
    encoded = base64.b64encode(content).decode("utf-8")

    # Determine MIME type
    img = Image.open(BytesIO(content))
    format_to_mime = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "BMP": "image/bmp",
    }
    mime_type = format_to_mime.get(img.format, "image/jpeg")

    return f"data:{mime_type};base64,{encoded}"

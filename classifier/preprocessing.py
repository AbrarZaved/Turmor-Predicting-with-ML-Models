"""
Preprocessing Module - OpenCV based, NO TensorFlow imports.

Optimized for Render Free Tier:
- Uses OpenCV (opencv-python-headless) for image processing
- Simple normalization (/255.0) instead of VGG16 preprocess_input
- Minimal memory footprint
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Image size must match model training
IMG_SIZE = 224


def preprocess_image(image_file):
    """
    Preprocess an uploaded image for model inference.

    Uses OpenCV for efficient image processing:
    1. Read image from file buffer
    2. Resize to 224x224
    3. Normalize to [0, 1] range
    4. Add batch dimension

    Args:
        image_file: Django UploadedFile or file-like object

    Returns:
        numpy.ndarray: Preprocessed image with shape (1, 224, 224, 3)
    """
    # Read file content into buffer
    if hasattr(image_file, "read"):
        file_bytes = image_file.read()
        # Reset file pointer for potential reuse
        if hasattr(image_file, "seek"):
            image_file.seek(0)
    else:
        with open(image_file, "rb") as f:
            file_bytes = f.read()

    # Decode image using OpenCV
    # cv2.IMREAD_COLOR ensures 3 channels (BGR)
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")

    # Convert BGR to RGB (OpenCV loads as BGR, model expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # Normalize to [0, 1] range
    image = image.astype(np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)

    return image


def get_original_image_base64(image_file):
    """
    Convert uploaded image to base64 for display in browser.

    Args:
        image_file: Django UploadedFile

    Returns:
        str: Base64 encoded image string with data URI prefix
    """
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

    # Determine MIME type using PIL
    img = Image.open(BytesIO(content))
    format_to_mime = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "BMP": "image/bmp",
    }
    mime_type = format_to_mime.get(img.format, "image/jpeg")

    return f"data:{mime_type};base64,{encoded}"

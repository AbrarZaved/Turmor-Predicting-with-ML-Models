"""
Model Loader Module - Singleton Pattern for TensorFlow Model.

This module provides a thread-safe, lazy-loading mechanism for the Keras model.
The model is loaded only once on first request and cached in memory.

Key Design Decisions:
- Lazy loading: Model not loaded until first prediction request
- Singleton: Only one model instance in memory
- Thread-safe: Uses threading.Lock for safe concurrent access
"""

import threading
from pathlib import Path
from django.conf import settings


class ModelLoader:
    """
    Singleton class for loading and caching the Keras model.

    Usage:
        model = ModelLoader.get_model()
        predictions = model.predict(preprocessed_image)
    """

    _model = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls):
        """
        Get the loaded Keras model. Loads on first call.

        Returns:
            tensorflow.keras.Model: The loaded brain tumor classification model

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model fails to load
        """
        # Fast path: model already loaded
        if cls._model is not None:
            return cls._model

        # Slow path: need to load model (thread-safe)
        with cls._lock:
            # Double-check after acquiring lock
            if cls._model is not None:
                return cls._model

            cls._model = cls._load_model()
            return cls._model

    @classmethod
    def _load_model(cls):
        """
        Internal method to load the Keras model from disk.

        Returns:
            The loaded Keras model
        """
        # Import TensorFlow here to avoid loading it during Django startup
        import tensorflow as tf

        # Limit TensorFlow threads for Render Free tier (low memory/CPU)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        model_path = Path(settings.MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please ensure '1_brain_tumor_vgg16.keras' is in the project root."
            )

        print(f"[ModelLoader] Loading model from: {model_path}")

        # Load the Keras model
        # compile=False since we only need inference, not training
        model = tf.keras.models.load_model(str(model_path), compile=False)

        # Warm up the model with a dummy prediction
        # This initializes all internal TensorFlow graphs for faster first real prediction
        import numpy as np

        dummy_input = np.zeros((1, settings.IMG_SIZE, settings.IMG_SIZE, 3))
        _ = model.predict(dummy_input, verbose=0)

        print(
            f"[ModelLoader] Model loaded successfully. Input shape: {model.input_shape}"
        )

        return model

    @classmethod
    def clear_model(cls):
        """
        Clear the cached model (useful for testing or memory management).
        """
        with cls._lock:
            cls._model = None

"""
Classifier app configuration.

This module handles:
1. App registration with Django
2. Lazy model loading on first request (not at startup)

The model is loaded lazily to avoid memory issues during migrations
and to ensure the model is only loaded when actually needed.
"""

from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    """Configuration for the classifier app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "classifier"
    verbose_name = "Brain Tumor Classifier"

    def ready(self):
        """
        Called when Django starts.

        We intentionally DO NOT load the model here because:
        1. It slows down Django startup (migrations, collectstatic, etc.)
        2. The model should be loaded only when first inference is needed

        Model loading is handled by the ModelLoader singleton in model_loader.py
        """
        pass

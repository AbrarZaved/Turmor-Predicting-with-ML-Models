"""
Django settings for brain_tumor_project.

Minimal configuration for medical image classification web app.
"""

import os
from pathlib import Path

# =============================================================================
# BASE CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: Keep the secret key secret in production!
SECRET_KEY = "django-insecure-change-this-in-production-abc123xyz"

# SECURITY WARNING: Don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "brain-tumor-predicting.onrender.com",
]


# =============================================================================
# APPLICATION DEFINITION
# =============================================================================
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    # Our classifier app
    "classifier",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "brain_tumor_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],  # App-level templates used
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.template.context_processors.media",
                "django.template.context_processors.static",
            ],
        },
    },
]

WSGI_APPLICATION = "brain_tumor_project.wsgi.application"


# =============================================================================
# DATABASE - Not required for this app, using minimal SQLite for Django internals
# =============================================================================
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# =============================================================================
# STATIC FILES (CSS, JavaScript, Images)
# =============================================================================
STATIC_URL = "/static/"
STATICFILES_DIRS = [
    BASE_DIR / "static",
]


# =============================================================================
# MEDIA FILES (User Uploads)
# =============================================================================
# This is where uploaded images will be stored
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Path to the trained Keras model file
MODEL_PATH = BASE_DIR.parent / "1_brain_tumor_vgg16.keras"

# Class labels matching the training data order
# These should match train_data.class_indices from your notebook
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "No Tumor",
    "pituitary",
]

# Image preprocessing settings (must match training)
IMG_SIZE = 224


# =============================================================================
# INTERNATIONALIZATION
# =============================================================================
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True


# =============================================================================
# DEFAULT PRIMARY KEY FIELD TYPE
# =============================================================================
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# =============================================================================
# FILE UPLOAD SETTINGS
# =============================================================================
# Maximum upload size: 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

# Allowed image extensions
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

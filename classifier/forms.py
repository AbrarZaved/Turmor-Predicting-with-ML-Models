"""
Django Forms for Image Upload.

Provides a clean, validated form for handling medical image uploads.
"""

from django import forms
from django.conf import settings
from pathlib import Path


class ImageUploadForm(forms.Form):
    """
    Form for uploading brain MRI images for tumor classification.

    Validates:
    - File is provided
    - File has valid image extension
    - File size is within limits
    """

    image = forms.ImageField(
        label="Upload Brain MRI Image",
        help_text="Supported formats: JPG, JPEG, PNG, BMP, GIF. Max size: 10MB",
        widget=forms.FileInput(
            attrs={
                "class": "form-control",
                "accept": "image/*",
            }
        ),
    )

    def clean_image(self):
        """
        Validate the uploaded image file.

        Returns:
            The validated image file

        Raises:
            ValidationError: If file is invalid
        """
        image = self.cleaned_data.get("image")

        if not image:
            raise forms.ValidationError("Please select an image file.")

        # Check file extension
        file_ext = Path(image.name).suffix.lower()
        allowed_extensions = getattr(
            settings,
            "ALLOWED_IMAGE_EXTENSIONS",
            [".jpg", ".jpeg", ".png", ".bmp", ".gif"],
        )

        if file_ext not in allowed_extensions:
            raise forms.ValidationError(
                f"Invalid file type '{file_ext}'. "
                f"Allowed types: {', '.join(allowed_extensions)}"
            )

        # Check file size (10MB max)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if image.size > max_size:
            raise forms.ValidationError(
                f"File too large. Maximum size is 10MB, "
                f"your file is {image.size / (1024*1024):.1f}MB"
            )

        return image

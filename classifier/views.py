"""
Views for Brain Tumor Classification.

Main view handles:
1. GET: Display upload form
2. POST: Process image, run inference, display results
"""

from django.shortcuts import render
from django.conf import settings

from .forms import ImageUploadForm
from .model_loader import ModelLoader
from .preprocessing import preprocess_image, get_original_image_base64


def predict_view(request):
    """
    Main prediction view.

    GET: Renders the upload form
    POST: Processes uploaded image and returns predictions

    Template context on POST:
        - form: The upload form (for re-submission)
        - image_data: Base64 encoded image for preview
        - prediction: Predicted class name
        - confidence: Confidence percentage (0-100)
        - all_predictions: List of (class_name, probability) tuples
        - error: Error message if something went wrong
    """
    context = {
        "form": ImageUploadForm(),
        "class_names": settings.CLASS_NAMES,
    }

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        context["form"] = form

        if form.is_valid():
            try:
                # Get uploaded file
                image_file = form.cleaned_data["image"]

                # Convert to base64 for preview display
                # (We don't save to disk to keep it minimal)
                image_data = get_original_image_base64(image_file)
                context["image_data"] = image_data
                context["image_name"] = image_file.name

                # Preprocess image for model
                preprocessed = preprocess_image(image_file)

                # Load model (lazy singleton - loads once, cached thereafter)
                model = ModelLoader.get_model()

                # Run inference
                # Output shape: (1, 4) - probabilities for each class
                predictions = model.predict(preprocessed, verbose=0)
                probabilities = predictions[0]  # Get first (and only) batch item

                # Get predicted class
                predicted_idx = int(probabilities.argmax())
                predicted_class = settings.CLASS_NAMES[predicted_idx]
                confidence = float(probabilities[predicted_idx]) * 100

                # Build all predictions list for display
                all_predictions = []
                for idx, class_name in enumerate(settings.CLASS_NAMES):
                    prob = float(probabilities[idx]) * 100
                    all_predictions.append(
                        {
                            "class_name": class_name,
                            "probability": prob,
                            "is_predicted": idx == predicted_idx,
                        }
                    )

                # Sort by probability descending
                all_predictions.sort(key=lambda x: x["probability"], reverse=True)

                # Add to context
                context["prediction"] = predicted_class
                context["confidence"] = confidence
                context["all_predictions"] = all_predictions
                context["success"] = True

            except FileNotFoundError as e:
                context["error"] = str(e)
            except Exception as e:
                context["error"] = f"Prediction failed: {str(e)}"
                # In production, log the full traceback
                import traceback

                print(f"[ERROR] Prediction error: {traceback.format_exc()}")

    return render(request, "classifier/predict.html", context)

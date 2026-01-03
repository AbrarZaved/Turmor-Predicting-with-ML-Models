# Brain Tumor Classification Web Application

A minimal Django-based web application for brain tumor MRI classification using a VGG16 transfer learning model.

## Features

- **Single Image Upload**: Upload brain MRI images via web interface
- **Real-time Classification**: Predicts tumor type with confidence score
- **Full Probability Distribution**: Shows probabilities for all 4 tumor classes
- **Clean UI**: Modern, responsive design with image preview

## Tumor Classes

The model classifies brain MRI images into 4 categories:

1. **Glioma** - Tumor originating from glial cells
2. **Meningioma** - Tumor arising from the meninges
3. **No Tumor** - Normal brain scan
4. **Pituitary** - Pituitary gland tumor

## Project Structure

```
brain_tumor_app/
├── brain_tumor_project/          # Django project settings
│   ├── settings.py               # Configuration (model path, class names)
│   ├── urls.py                   # URL routing
│   └── wsgi.py                   # WSGI application
├── classifier/                   # Django app
│   ├── apps.py                   # App configuration
│   ├── forms.py                  # Image upload form
│   ├── model_loader.py           # Lazy model loading singleton
│   ├── preprocessing.py          # Image preprocessing pipeline
│   ├── views.py                  # Main prediction view
│   ├── urls.py                   # App URL patterns
│   └── templates/
│       └── classifier/
│           └── predict.html      # Web interface
├── media/                        # (auto-created) Uploaded files
├── static/                       # Static files
├── manage.py                     # Django management script
└── requirements.txt              # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10+
- pip (Python package manager)

### Setup

1. **Navigate to the project directory:**

   ```bash
   cd brain_tumor_app
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model file is in place:**

   The model file `1_brain_tumor_vgg16.keras` should be in the parent directory:

   ```
   Raha/
   ├── 1_brain_tumor_vgg16.keras    # <-- Model file here
   └── brain_tumor_app/
       └── ...
   ```

5. **Run the development server:**

   ```bash
   python manage.py runserver
   ```

6. **Open in browser:**
   ```
   http://127.0.0.1:8000/
   ```

## Usage

1. Click "Choose Image File" to select a brain MRI image
2. Click "Analyze Image" to run the classification
3. View results:
   - Uploaded image preview
   - Predicted tumor class
   - Confidence percentage
   - Full probability distribution for all classes

## Configuration

### Model Path

Edit `brain_tumor_project/settings.py` if your model is in a different location:

```python
MODEL_PATH = BASE_DIR.parent / '1_brain_tumor_vgg16.keras'
```

### Class Names

If your model uses different class names, update:

```python
CLASS_NAMES = [
    'glioma',
    'meningioma',
    'notumor',
    'pituitary',
]
```

**Important**: The order must match `train_data.class_indices` from training.

## Technical Details

### Model Loading Strategy

- **Lazy Loading**: Model is loaded on first prediction request, not at startup
- **Singleton Pattern**: Only one model instance in memory
- **Thread-Safe**: Uses threading locks for concurrent requests
- **Warm-up**: Initial dummy prediction to initialize TensorFlow graphs

### Preprocessing Pipeline

Matches the training preprocessing exactly:

1. Read image using Pillow
2. Convert to RGB (3 channels)
3. Resize to 224x224 (VGG16 input size)
4. Convert to numpy array
5. Apply VGG16 preprocessing:
   - RGB to BGR conversion
   - ImageNet mean subtraction

### Performance Considerations

- First request takes longer (model loading + TensorFlow initialization)
- Subsequent requests are fast (model cached in memory)
- Model uses ~500MB RAM
- Inference takes ~100-500ms depending on hardware

## Troubleshooting

### Model not found error

Ensure `1_brain_tumor_vgg16.keras` is in the correct location (parent of `brain_tumor_app`).

### TensorFlow GPU issues

If you have GPU issues, force CPU:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Memory errors

The VGG16 model requires ~500MB RAM. Ensure sufficient memory available.

## License

For educational and research purposes.

"""
URL configuration for brain_tumor_project.

Routes all requests to the classifier app.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

urlpatterns = [
    # Main classifier app - handles image upload and prediction
    path("", include("classifier.urls")),
]

# Serve media files during development
# In production, use a proper web server (nginx, etc.) to serve media
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

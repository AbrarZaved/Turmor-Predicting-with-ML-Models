"""
URL patterns for the classifier app.
"""

from django.urls import path
from . import views

app_name = "classifier"

urlpatterns = [
    # Main prediction page (also serves as homepage)
    path("", views.predict_view, name="predict"),
]


from .views import Predict
from django.urls import path

urlpatterns = [
    path('predict', Predict.as_view(), name='predict'),
]

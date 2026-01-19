from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/heart/', views.predict_heart, name='predict_heart'),
    path('predict/diabetes/', views.predict_diabetes_view, name='predict_diabetes'),
    path('predict/cancer/', views.predict_cancer_view, name='predict_cancer'),
    path('train-models/', views.train_models, name='train_models'),
]

from django.shortcuts import render, redirect
from django.http import JsonResponse
from .ml_utils import predict_heart_disease, predict_diabetes, predict_cancer
import json

def get_required_float(request, field_name):
    val = request.POST.get(field_name)
    if not val or val.strip() == '':
        raise ValueError(f"{field_name.replace('_', ' ').title()} is required.")
    return float(val)

def home(request):
    return render(request, 'predictor/home.html')

def predict_heart(request):
    if request.method == 'POST':
        try:
            # Get form data
            age = get_required_float(request, 'age')
            sex = get_required_float(request, 'sex')
            cp = get_required_float(request, 'cp')
            trestbps = get_required_float(request, 'trestbps')
            chol = get_required_float(request, 'chol')
            fbs = get_required_float(request, 'fbs')
            restecg = get_required_float(request, 'restecg')
            thalach = get_required_float(request, 'thalach')
            exang = get_required_float(request, 'exang')
            oldpeak = get_required_float(request, 'oldpeak')
            slope = get_required_float(request, 'slope')
            ca = get_required_float(request, 'ca')
            thal = get_required_float(request, 'thal')
            
            # Make prediction
            prediction, probability, feature_importance = predict_heart_disease(
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal
            )

            # Format feature importance for template
            formatted_feature_importance = {}
            for name, imp in feature_importance.items():
                formatted_feature_importance[name] = {
                    'value': imp,
                    'width': f"{int(imp * 100)}%"
                }

            return render(request, 'predictor/result.html', {
                'disease': 'Heart Disease',
                'prediction': prediction,
                'probability': probability,
                'feature_importance': formatted_feature_importance,
                'input_data': {
                    'Age': age, 'Sex': sex, 'Chest Pain Type': cp,
                    'Resting BP': trestbps, 'Cholesterol': chol,
                    'Fasting Blood Sugar': fbs, 'Resting ECG': restecg,
                    'Max Heart Rate': thalach, 'Exercise Angina': exang,
                    'ST Depression': oldpeak, 'Slope': slope,
                    'Major Vessels': ca, 'Thal': thal
                }
            })
            
        except Exception as e:
            return render(request, 'predictor/predict_heart.html', 
                         {'error': f'Error: {str(e)}'})
    
    return render(request, 'predictor/predict_heart.html')

def predict_diabetes_view(request):
    if request.method == 'POST':
        try:
            # Get form data
            pregnancies = get_required_float(request, 'pregnancies')
            glucose = get_required_float(request, 'glucose')
            blood_pressure = get_required_float(request, 'blood_pressure')
            skin_thickness = get_required_float(request, 'skin_thickness')
            insulin = get_required_float(request, 'insulin')
            bmi = get_required_float(request, 'bmi')
            diabetes_pedigree = get_required_float(request, 'diabetes_pedigree')
            age = get_required_float(request, 'age')
            
            # Make prediction
            prediction, probability = predict_diabetes(
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age
            )
            
            return render(request, 'predictor/result.html', {
                'disease': 'Diabetes',
                'prediction': prediction,
                'probability': probability,
                'input_data': {
                    'Pregnancies': pregnancies, 'Glucose': glucose,
                    'Blood Pressure': blood_pressure, 'Skin Thickness': skin_thickness,
                    'Insulin': insulin, 'BMI': bmi,
                    'Diabetes Pedigree': diabetes_pedigree, 'Age': age
                }
            })
            
        except Exception as e:
            return render(request, 'predictor/predict_diabetes.html', 
                         {'error': f'Error: {str(e)}'})
    
    return render(request, 'predictor/predict_diabetes.html')

def predict_cancer_view(request):
    if request.method == 'POST':
        try:
            # Get form data (using mean radius, texture, perimeter, area, smoothness)
            mean_radius = get_required_float(request, 'mean_radius')
            mean_texture = get_required_float(request, 'mean_texture')
            mean_perimeter = get_required_float(request, 'mean_perimeter')
            mean_area = get_required_float(request, 'mean_area')
            mean_smoothness = get_required_float(request, 'mean_smoothness')
            
            # Make prediction
            prediction, probability = predict_cancer(
                mean_radius, mean_texture, mean_perimeter, 
                mean_area, mean_smoothness
            )
            
            return render(request, 'predictor/result.html', {
                'disease': 'Breast Cancer',
                'prediction': prediction,
                'probability': probability,
                'input_data': {
                    'Mean Radius': mean_radius, 'Mean Texture': mean_texture,
                    'Mean Perimeter': mean_perimeter, 'Mean Area': mean_area,
                    'Mean Smoothness': mean_smoothness
                }
            })
            
        except Exception as e:
            return render(request, 'predictor/predictt_cancer.html',
                         {'error': f'Error: {str(e)}'})

    return render(request, 'predictor/predictt_cancer.html')

def train_models(request):
    from .ml_utils import train_all_models
    if request.method == 'POST':
        try:
            accuracy_scores = train_all_models()
            return JsonResponse({'success': True, 'accuracy': accuracy_scores})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return render(request, 'predictor/train_models.html')
# Create your views here.

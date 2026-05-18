import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from .data_loader import load_heart_data, load_diabetes_data, load_cancer_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml_models')
SCALERS_DIR = os.path.join(MODELS_DIR, 'scalers')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)

def train_heart_model():
    """Train heart disease prediction model"""
    X, y = load_heart_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, os.path.join(MODELS_DIR, 'heart_model.pkl'))
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def train_diabetes_model():
    """Train diabetes prediction model"""
    X, y = load_diabetes_data()
    
    # Handle missing values
    X = np.nan_to_num(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, os.path.join(MODELS_DIR, 'diabetes_model.pkl'))
    joblib.dump(scaler, os.path.join(SCALERS_DIR, 'diabetes_scaler.pkl'))
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def train_cancer_model():
    """Train breast cancer prediction model"""
    X, y = load_cancer_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, os.path.join(MODELS_DIR, 'cancer_model.pkl'))
    joblib.dump(scaler, os.path.join(SCALERS_DIR, 'cancer_scaler.pkl'))
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def train_all_models():
    """Train all models"""
    accuracies = {}
    
    print("Training Heart Disease Model...")
    accuracies['heart'] = train_heart_model()
    
    print("Training Diabetes Model...")
    accuracies['diabetes'] = train_diabetes_model()
    
    print("Training Cancer Model...")
    accuracies['cancer'] = train_cancer_model()
    
    print(f"Training complete! Accuracies: {accuracies}")
    return accuracies

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, 
                         thalach, exang, oldpeak, slope, ca, thal):
    """Predict heart disease"""
    model_path = os.path.join(MODELS_DIR, 'heart_model.pkl')
    
    if not os.path.exists(model_path):
        train_heart_model()
    
    model = joblib.load(model_path)
    
    # Prepare input
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    
    # Get feature importance
    feature_names = ['Age', 'Sex', 'Chest Pain', 'Resting BP', 'Cholesterol',
                    'Fasting BS', 'Resting ECG', 'Max Heart Rate',
                    'Exercise Angina', 'ST Depression', 'Slope',
                    'Major Vessels', 'Thal']
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    return "Positive" if prediction == 1 else "Negative", round(probability, 2), feature_importance

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                    insulin, bmi, diabetes_pedigree, age):
    """Predict diabetes"""
    model_path = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
    scaler_path = os.path.join(SCALERS_DIR, 'diabetes_scaler.pkl')
    
    if not os.path.exists(model_path):
        train_diabetes_model()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100
    
    return "Positive" if prediction == 1 else "Negative", round(probability, 2)

def predict_cancer(mean_radius, mean_texture, mean_perimeter, 
                  mean_area, mean_smoothness):
    """Predict breast cancer"""
    model_path = os.path.join(MODELS_DIR, 'cancer_model.pkl')
    scaler_path = os.path.join(SCALERS_DIR, 'cancer_scaler.pkl')
    
    if not os.path.exists(model_path):
        train_cancer_model()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare input (using first 5 features for simplicity)
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter,
                          mean_area, mean_smoothness]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100
    
    return "Malignant" if prediction == 1 else "Benign", round(probability, 2)
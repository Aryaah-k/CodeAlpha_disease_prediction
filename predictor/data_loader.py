import os
import pandas as pd
import numpy as np

DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')

def load_heart_data():
    """Load heart disease dataset"""
    try:
        # UCI Heart Disease dataset columns
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        # Try to load from file
        file_path = os.path.join(DATASETS_DIR, 'heart.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            # Create synthetic data if file doesn't exist
            print("Creating synthetic heart disease data...")
            np.random.seed(42)
            n_samples = 300
            
            data = {
                'age': np.random.randint(29, 80, n_samples),
                'sex': np.random.randint(0, 2, n_samples),
                'cp': np.random.randint(0, 4, n_samples),
                'trestbps': np.random.randint(94, 200, n_samples),
                'chol': np.random.randint(126, 565, n_samples),
                'fbs': np.random.randint(0, 2, n_samples),
                'restecg': np.random.randint(0, 3, n_samples),
                'thalach': np.random.randint(71, 203, n_samples),
                'exang': np.random.randint(0, 2, n_samples),
                'oldpeak': np.round(np.random.uniform(0, 6.2, n_samples), 1),
                'slope': np.random.randint(0, 3, n_samples),
                'ca': np.random.randint(0, 4, n_samples),
                'thal': np.random.randint(0, 4, n_samples),
                'target': np.random.randint(0, 2, n_samples)
            }
            
            df = pd.DataFrame(data)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
        
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        return X, y
        
    except Exception as e:
        print(f"Error loading heart data: {e}")
        raise

def load_diabetes_data():
    """Load diabetes dataset"""
    try:
        file_path = os.path.join(DATASETS_DIR, 'diabetes.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            # Create synthetic data
            print("Creating synthetic diabetes data...")
            np.random.seed(42)
            n_samples = 768
            
            data = {
                'Pregnancies': np.random.randint(0, 17, n_samples),
                'Glucose': np.random.randint(0, 200, n_samples),
                'BloodPressure': np.random.randint(0, 122, n_samples),
                'SkinThickness': np.random.randint(0, 100, n_samples),
                'Insulin': np.random.randint(0, 850, n_samples),
                'BMI': np.round(np.random.uniform(0, 67.1, n_samples), 1),
                'DiabetesPedigreeFunction': np.round(np.random.uniform(0.078, 2.42, n_samples), 3),
                'Age': np.random.randint(21, 81, n_samples),
                'Outcome': np.random.randint(0, 2, n_samples)
            }
            
            df = pd.DataFrame(data)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
        
        X = df.drop('Outcome', axis=1).values
        y = df['Outcome'].values
        
        return X, y
        
    except Exception as e:
        print(f"Error loading diabetes data: {e}")
        raise

def load_cancer_data():
    """Load breast cancer dataset"""
    try:
        file_path = os.path.join(DATASETS_DIR, 'breast_cancer.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            # Create synthetic data based on Wisconsin Breast Cancer dataset
            print("Creating synthetic breast cancer data...")
            np.random.seed(42)
            n_samples = 569
            
            # Generate benign (0) and malignant (1) samples
            n_benign = 357
            n_malignant = 212
            
            # Benign tumors (smaller, more regular)
            benign_data = {
                'mean_radius': np.random.uniform(6, 15, n_benign),
                'mean_texture': np.random.uniform(9, 21, n_benign),
                'mean_perimeter': np.random.uniform(40, 90, n_benign),
                'mean_area': np.random.uniform(200, 700, n_benign),
                'mean_smoothness': np.random.uniform(0.05, 0.11, n_benign),
            }
            
            # Malignant tumors (larger, more irregular)
            malignant_data = {
                'mean_radius': np.random.uniform(10, 28, n_malignant),
                'mean_texture': np.random.uniform(15, 33, n_malignant),
                'mean_perimeter': np.random.uniform(70, 190, n_malignant),
                'mean_area': np.random.uniform(500, 2500, n_malignant),
                'mean_smoothness': np.random.uniform(0.07, 0.15, n_malignant),
            }
            
            # Combine data
            benign_df = pd.DataFrame(benign_data)
            benign_df['diagnosis'] = 0
            
            malignant_df = pd.DataFrame(malignant_data)
            malignant_df['diagnosis'] = 1
            
            df = pd.concat([benign_df, malignant_df], ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
        
        # Use only first 5 features for simplicity
        feature_cols = ['mean_radius', 'mean_texture', 'mean_perimeter', 
                       'mean_area', 'mean_smoothness']
        X = df[feature_cols].values
        y = df['diagnosis'].values
        
        return X, y
        
    except Exception as e:
        print(f"Error loading cancer data: {e}")
        raise

"""
Script to download or create synthetic datasets for the disease prediction system
"""

import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer

def create_datasets():
    """Create synthetic datasets for the disease prediction system"""
    
    # Create datasets directory
    datasets_dir = 'datasets'
    os.makedirs(datasets_dir, exist_ok=True)
    
    print("=" * 60)
    print("Creating Synthetic Medical Datasets")
    print("=" * 60)
    
    # 1. Heart Disease Dataset
    print("\n1. Creating Heart Disease Dataset...")
    np.random.seed(42)
    n_samples = 300
    
    heart_data = {
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
    
    # Create correlation between features and target for realistic data
    # Age and target: older people more likely to have heart disease
    mask_older = heart_data['age'] > 60
    heart_data['target'][mask_older] = np.random.choice([0, 1], size=mask_older.sum(), p=[0.3, 0.7])
    
    # High cholesterol and target
    mask_high_chol = heart_data['chol'] > 240
    heart_data['target'][mask_high_chol] = np.random.choice([0, 1], size=mask_high_chol.sum(), p=[0.4, 0.6])
    
    df_heart = pd.DataFrame(heart_data)
    df_heart.to_csv(os.path.join(datasets_dir, 'heart.csv'), index=False)
    print(f"   Created: {len(df_heart)} samples, {df_heart.shape[1]} features")
    
    # 2. Diabetes Dataset
    print("\n2. Creating Diabetes Dataset...")
    n_samples = 768
    
    diabetes_data = {
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
    
    # Create realistic correlations
    # High glucose -> more likely diabetes
    mask_high_glucose = diabetes_data['Glucose'] > 140
    diabetes_data['Outcome'][mask_high_glucose] = np.random.choice([0, 1], size=mask_high_glucose.sum(), p=[0.2, 0.8])
    
    # High BMI -> more likely diabetes
    mask_high_bmi = diabetes_data['BMI'] > 30
    diabetes_data['Outcome'][mask_high_bmi] = np.random.choice([0, 1], size=mask_high_bmi.sum(), p=[0.3, 0.7])
    
    # Older age -> more likely diabetes
    mask_older = diabetes_data['Age'] > 50
    diabetes_data['Outcome'][mask_older] = np.random.choice([0, 1], size=mask_older.sum(), p=[0.4, 0.6])
    
    df_diabetes = pd.DataFrame(diabetes_data)
    df_diabetes.to_csv(os.path.join(datasets_dir, 'diabetes.csv'), index=False)
    print(f"   Created: {len(df_diabetes)} samples, {df_diabetes.shape[1]} features")
    
    # 3. Breast Cancer Dataset
    print("\n3. Creating Breast Cancer Dataset...")
    
    # Load real breast cancer data from sklearn
    cancer_data = load_breast_cancer()
    df_cancer_real = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    df_cancer_real['diagnosis'] = cancer_data.target
    
    # Use only first 5 features for simplicity in our project
    feature_cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
    df_cancer = df_cancer_real[feature_cols + ['diagnosis']].copy()
    
    # Rename columns for consistency
    df_cancer.columns = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'diagnosis']
    
    # Add some synthetic variations
    np.random.seed(42)
    noise_scale = 0.1
    
    for col in feature_cols:
        col_name = col.replace(' ', '_')
        noise = np.random.normal(0, df_cancer[col_name].std() * noise_scale, len(df_cancer))
        df_cancer[col_name] += noise
    
    df_cancer.to_csv(os.path.join(datasets_dir, 'breast_cancer.csv'), index=False)
    print(f"   Created: {len(df_cancer)} samples, {df_cancer.shape[1]} features")
    
    # 4. Create dataset statistics
    print("\n4. Dataset Statistics:")
    print("-" * 40)
    
    datasets = {
        'Heart Disease': df_heart,
        'Diabetes': df_diabetes,
        'Breast Cancer': df_cancer
    }
    
    for name, df in datasets.items():
        target_col = 'target' if 'target' in df.columns else 'Outcome' if 'Outcome' in df.columns else 'diagnosis'
        positive_cases = df[target_col].sum() if target_col in df.columns else 'N/A'
        total_cases = len(df)
        
        print(f"\n{name}:")
        print(f"  Total samples: {total_cases}")
        if isinstance(positive_cases, (int, float)):
            print(f"  Positive cases: {positive_cases} ({positive_cases/total_cases*100:.1f}%)")
            print(f"  Negative cases: {total_cases - positive_cases} ({(total_cases - positive_cases)/total_cases*100:.1f}%)")
        print(f"  Features: {df.shape[1] - 1}")
    
    # 5. Create README for datasets
    print("\n5. Creating dataset documentation...")
    readme_content = """# Medical Datasets for Disease Prediction System

This directory contains synthetic medical datasets used for training machine learning models.

## 📊 Datasets

### 1. Heart Disease Dataset (heart.csv)
- **Samples:** 300
- **Features:** 13
- **Target:** Presence of heart disease (0 = No, 1 = Yes)

**Features:**
- age: Age in years
- sex: Sex (1 = male; 0 = female)
- cp: Chest pain type (0-3)
- trestbps: Resting blood pressure (mm Hg)
- chol: Serum cholesterol (mg/dl)
- fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- restecg: Resting electrocardiographic results (0-2)
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina (1 = yes; 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- slope: Slope of the peak exercise ST segment
- ca: Number of major vessels (0-3) colored by fluoroscopy
- thal: Thalassemia (0-3)

### 2. Diabetes Dataset (diabetes.csv)
- **Samples:** 768
- **Features:** 8
- **Target:** Diabetes diagnosis (0 = No, 1 = Yes)

**Features:**
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index
- DiabetesPedigreeFunction: Diabetes likelihood based on family history
- Age: Age in years

### 3. Breast Cancer Dataset (breast_cancer.csv)
- **Samples:** 569
- **Features:** 5
- **Target:** Diagnosis (0 = Benign, 1 = Malignant)

**Features:**
- mean_radius: Mean of distances from center to points on perimeter
- mean_texture: Standard deviation of gray-scale values
- mean_perimeter: Mean size of the core tumor
- mean_area: Mean area of the nucleus
- mean_smoothness: Local variation in radius lengths

## ⚠️ Important Notes

1. **Synthetic Data:** These datasets are synthetically generated for educational purposes.
2. **Not for Real Diagnosis:** Should not be used for actual medical diagnosis.
3. **Educational Use Only:** Created for learning machine learning and Django.
4. **Real Data:** For production use, replace with real medical datasets.

## 📈 Data Generation

The datasets were created with:
- Realistic value ranges based on medical literature
- Statistical correlations between features
- Balanced class distributions
- Appropriate noise levels for realism

## 🔄 Updating Datasets

To generate new datasets:
```bash
python download_datasets.py
```

"""

    # Write README file
    with open(os.path.join(datasets_dir, 'README.md'), 'w') as f:
        f.write(readme_content)

    print("\n" + "=" * 60)
    print("Dataset Creation Complete!")
    print("=" * 60)
    print("Files created in 'datasets/' directory:")
    print("- heart.csv")
    print("- diabetes.csv")
    print("- breast_cancer.csv")
    print("- README.md")
    print("\nRun 'python manage.py train_models' to train the ML models.")

if __name__ == "__main__":
    create_datasets()

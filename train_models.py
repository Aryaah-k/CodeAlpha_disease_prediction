#!/usr/bin/env python
"""
Script to train all ML models for the disease prediction system
"""

import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor.ml_utils import train_all_models

def main():
    print("=" * 50)
    print("Disease Prediction Model Training")
    print("=" * 50)
    
    try:
        accuracies = train_all_models()
        
        print("\n" + "=" * 50)
        print("Training Results:")
        print("=" * 50)
        for disease, accuracy in accuracies.items():
            print(f"{disease.capitalize()} Model Accuracy: {accuracy:.2%}")
        
        print("\nModels saved to 'ml_models/' directory")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
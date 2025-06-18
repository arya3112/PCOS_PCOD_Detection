import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    # Load model and scaler
    print("Loading model files...")
    model = joblib.load('pcos_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    print(f"Model type: {type(model)}")
    print(f"Scaler type: {type(scaler)}")
    
    # Test prediction
    print("\nTesting prediction...")
    test_data = pd.DataFrame({
        'age': [25],
        'bmi': [22.0],
        'cycle_regularity': [3],
        'weight_gain': [2],
        'hair_growth': [1],
        'skin_darkening': [1],
        'hair_loss': [1],
        'pimples': [1],
        'fast_food': [2],
        'exercise': [3]
    })
    
    # Scale the data
    test_scaled = scaler.transform(test_data)
    
    # Make prediction
    prediction = model.predict_proba(test_scaled)[0]
    probability = prediction[1] * 100
    
    print(f"Prediction successful!")
    print(f"Risk probability: {probability:.2f}%")
    print(f"Risk level: {'High Risk' if probability > 50 else 'Low Risk'}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print(f"Error type: {type(e)}") 
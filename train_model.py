import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Define feature names
FEATURE_NAMES = [
    'age', 'bmi', 'cycle_regularity', 'weight_gain', 'hair_growth',
    'skin_darkening', 'hair_loss', 'pimples', 'fast_food', 'exercise'
]

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    age = np.random.uniform(15, 45, n_samples)
    bmi = np.random.uniform(18, 40, n_samples)
    cycle_regularity = np.random.randint(1, 6, n_samples)
    weight_gain = np.random.randint(1, 6, n_samples)
    hair_growth = np.random.randint(1, 6, n_samples)
    skin_darkening = np.random.randint(1, 6, n_samples)
    hair_loss = np.random.randint(1, 6, n_samples)
    pimples = np.random.randint(1, 6, n_samples)
    fast_food = np.random.randint(1, 6, n_samples)
    exercise = np.random.randint(1, 6, n_samples)
    
    # Create features DataFrame with proper feature names
    features = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'cycle_regularity': cycle_regularity,
        'weight_gain': weight_gain,
        'hair_growth': hair_growth,
        'skin_darkening': skin_darkening,
        'hair_loss': hair_loss,
        'pimples': pimples,
        'fast_food': fast_food,
        'exercise': exercise
    }, columns=FEATURE_NAMES)
    
    # Generate labels based on feature combinations
    # Higher risk for:
    # - Higher BMI
    # - Irregular cycles
    # - More symptoms (hair growth, skin darkening, etc.)
    # - Less exercise
    # - More fast food consumption
    
    risk_score = (
        (bmi - 18) / 22 * 0.2 +  # BMI contribution
        (6 - cycle_regularity) / 5 * 0.2 +  # Cycle irregularity contribution
        (hair_growth + skin_darkening + hair_loss + pimples) / 20 * 0.3 +  # Symptoms contribution
        (6 - exercise) / 5 * 0.15 +  # Exercise contribution
        fast_food / 5 * 0.15  # Fast food contribution
    )
    
    labels = (risk_score > 0.5).astype(int)
    
    return features, labels

def train_model():
    # Generate synthetic data
    X, y = generate_synthetic_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames with feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURE_NAMES)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURE_NAMES)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled_df, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train_scaled_df, y_train)
    test_score = model.score(X_test_scaled_df, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")
    
    # Save the model and scaler
    joblib.dump(model, 'pcos_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Save feature names
    with open('feature_names.json', 'w') as f:
        json.dump(FEATURE_NAMES, f)
    
    print("Model, scaler, and feature names saved successfully!")

if __name__ == "__main__":
    train_model() 
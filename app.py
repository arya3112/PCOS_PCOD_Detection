from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
model = None
try:
    model = joblib.load('pcos_model.joblib')
except:
    print("Model file not found. Please train the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get values from the form
        data = {
            'age': float(request.form['age']),
            'bmi': float(request.form['bmi']),
            'cycle_regularity': float(request.form['cycle_regularity']),
            'weight_gain': float(request.form['weight_gain']),
            'hair_growth': float(request.form['hair_growth']),
            'skin_darkening': float(request.form['skin_darkening']),
            'hair_loss': float(request.form['hair_loss']),
            'pimples': float(request.form['pimples']),
            'fast_food': float(request.form['fast_food']),
            'exercise': float(request.form['exercise'])
        }

        # Convert to numpy array and reshape
        features = np.array([[
            data['age'], data['bmi'], data['cycle_regularity'],
            data['weight_gain'], data['hair_growth'], data['skin_darkening'],
            data['hair_loss'], data['pimples'], data['fast_food'],
            data['exercise']
        ]])

        # Make prediction
        prediction = model.predict_proba(features)[0]
        probability = prediction[1] * 100

        result = {
            'prediction': 'High Risk' if probability > 50 else 'Low Risk',
            'probability': round(probability, 2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 
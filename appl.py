# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load(open('final_model.joblib', 'rb'))


# Define Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        age = float(data.get('age', 0))
        stress_level = float(data.get('stressLevel', 0))
        heart_rate = float(data.get('heartRate', 0))
        occupation = int(data.get('occupation',0))
        bmi_category = int(data.get('bmiCategory',0))
        blood_pressure = int (data.get('bloodPressure',0))
        # Make predictions using the loaded model
        features = np.array([age, occupation, stress_level, bmi_category, heart_rate, blood_pressure]).reshape(1, -1)
        predictions = model.predict(features)

        # Convert predictions to list for JSON response
        predictions_list = predictions.tolist()

        return jsonify({'prediction': predictions_list[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
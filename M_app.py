from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from joblib import load
import pandas as pd
import requests
import requests 
import json


app = Flask(__name__)

# Load your model
model_path = r"C:\Users\user\Desktop\BRCA\random_forest_model.joblib"
rf_model = load(model_path)
print("Random Forest Model Loaded:", rf_model)

# Standard
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received Data:", data)
    if not data:
        return jsonify({"error": "Invalid input"}), 400
    df = pd.DataFrame(data)
    print("DataFrame Created:", df.head())
    print("DataFrame Columns:", df.columns)
    #predict 
    predictions = rf_model.predict(df)
    return jsonify(predictions.tolist())



if __name__ == '__main__':
    app.run(debug=False)

# For testing the endpoint:
data = { 
    "cn_A2ML1": [1],
    "cn_ABAT": [0.5], 
    "cn_ABCA10": [0.3], 
    "cn_ABCA4": [0.8], 
    "cn_ABCA6": [1], 
    "cn_ANKRD30B": [1], 
    "rs_APOB": [0.5], 
    "rs_KRT23": [0.3], 
    "mu_TNXB": [0.8],
} 
headers = { 
    'Content-Type': 'application/json' 
} 
response = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(data), headers=headers) 
print(response.json())

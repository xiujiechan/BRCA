from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import feyn
import pandas as pd
import requests
import json
import pickle

app = Flask(__name__)

# Load your model
model_path = r"C:\Users\user\Desktop\archive\model1_200its_mc7.model"
multi_model = feyn.Model.load(model_path)
print("Model Loaded:", multi_model)

# Save model
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

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
    predictions = multi_model.predict(df)
    return jsonify(predictions.tolist())



if __name__ == '__main__':
    app.run(debug=True)

# For testing the endpoint:
data = {
    "feature1": [1, 2, 3],
    "feature2": [4, 5, 6]
}

headers = {
    'Content-Type': 'application/json'
}

{
    "cn_ANKRD30B": [1],
    "rs_APOB": [0.5],
    "rs_KRT23":[0.3],
    "mu_TNXB":[0.8],
    "feature1": [1],
    "feature2": [2]
}

response = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(data), headers=headers)
print(response.json())

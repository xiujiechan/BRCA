from flask import Flask, request, jsonify
import feyn
import pandas as pd

app = Flask(__name__)

# Load your model
model_path = r"C:\Users\user\Desktop\archive\model1_200its_mc7.model"
multi_model = feyn.Model.load(model_path)
print("Model Loaded:", multi_model)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Here you process the incoming data and use the model to make predictions
    # Example: 
    df = pd.DataFrame(data)
    predictions = multi_model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)

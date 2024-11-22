from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the saved XGBoost pipeline model
try:
    with open("xgboost_apple_quality_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Model file not found. Ensure 'xgboost_apple_quality_model.pkl' is in the same directory.")

# Initialize the Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Apple Quality Prediction API!"

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "This endpoint expects POST requests.", 405

    try:
        # Parse the input JSON data
        data = request.get_json()

        # Ensure the data contains the 'features' key
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in input JSON"}), 400

        # Convert features to a NumPy array
        features = np.array(data["features"]).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()

        # Prepare the response
        response = {
            "prediction": int(prediction),  # 1 for good, 0 for bad
            "probability": probability
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Set the host to 0.0.0.0 for compatibility
    app.run(host="127.0.0.1", port=5000, debug=True)

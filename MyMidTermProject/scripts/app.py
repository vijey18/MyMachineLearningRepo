from flask import Flask, jsonify, request
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the model
try:
    with open("xgboost_apple_quality_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Model file not found. Ensure 'xgboost_model.pkl' is in the same directory.")

# Define the root endpoint ("/")
@app.route("/", methods=['GET'])
def home():
    return "Welcome to the Apple Quality Prediction API!"

# Define the /predict route that supports both GET and POST methods
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
                return """
        <h1>Prediction Endpoint</h1>
        <p>This endpoint expects POST requests with JSON data containing a 'features' key, please use curl to test the same</p>
        <p>Example:</p>
        <pre>{
            "features": [-3.970049, -2.512336, 5.346330, -1.012009, 1.844900, 0.329840, -0.491590]
        }</pre>
        """, 200
    
    if request.method == 'POST':
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
    # Set the host to 0.0.0.0 for compatibility with Docker
    app.run(host="0.0.0.0", port=5000, debug=True)

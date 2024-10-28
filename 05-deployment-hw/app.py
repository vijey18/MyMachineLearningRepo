from flask import Flask, request, jsonify
import pickle

# Load the model and DictVectorizer
with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)
with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()  # Get client data from the POST request
    X = dv.transform([client])   # Transform client data with DictVectorizer
    prob = model.predict_proba(X)[0, 1]  # Get the probability
    return jsonify({'subscription_probability': prob})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9696)

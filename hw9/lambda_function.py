from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the model
MODEL_PATH = "./model_2024_hairstyle_v2.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocessing function
def preprocess_image(image_path):
    target_size = (150, 150)  # Same size used in homework 8
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension
    return img_array

# Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read image file
        file = request.files["image"]
        img_path = "./score_image.jpeg"
        file.save(img_path)
        
        # Preprocess the image
        input_data = preprocess_image(img_path)
        
        # Perform inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])
        
        # Return the prediction
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

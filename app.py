from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow Flutter app to access

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'Abdullah.h5')

# WeTransfer temporary model download link
MODEL_URL = "https://we.tl/t-sveBQTiV9B"  # 🔁 Replace if this expires

# Download the model if not present
model = None
model_load_error = None

if not os.path.exists(MODEL_PATH):
    try:
        print("⬇️ Downloading model from WeTransfer...")
        response = requests.get(MODEL_URL, allow_redirects=True)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("✅ Model downloaded.")
    except Exception as e:
        model_load_error = f"Download failed: {str(e)}"
        print(model_load_error)

# Try to load the model
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully.")
    except Exception as e:
        model_load_error = str(e)
        model = None
        print("❌ Error loading model:", e)
else:
    model_load_error = "Model file was not downloaded successfully."

# Define tumor classes
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Model could not be loaded on server.',
            'details': model_load_error
        }), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Save image temporarily
    uploads_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    image_path = os.path.join(uploads_dir, file.filename)
    file.save(image_path)

    try:
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        label = class_labels[class_index]

        return jsonify({
            'label': label,
            'confidence': round(confidence * 100, 2),
            'message': 'No Tumor Detected' if label == 'notumor' else f'Tumor Detected: {label}'
        })
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

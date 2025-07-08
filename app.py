from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow Flutter app to access

# Get absolute path to the model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'Abdullah.h5')

# === Try safe loading with compile=False to avoid deserialization issues ===
model = None
model_load_error = None
if not os.path.exists(MODEL_PATH):
    model_load_error = f"Model file not found at {MODEL_PATH}"
    print(model_load_error)
else:
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        model_load_error = str(e)
        print("Error loading model:", e)
        model = None

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model could not be loaded on server.', 'details': model_load_error}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Ensure 'uploads' folder exists
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
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

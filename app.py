# app.py
# Flask app to serve a trained plant disease CNN model

import os
import io
import pickle
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import model_from_json

# ========================
# Configuration
# ========================
IMAGE_SIZE = 224  # must match your training image size
MODEL_ARCH_PATH = 'plant_disease_model_architecture.pkl'
MODEL_WEIGHTS_PATH = 'plant_disease_model_weights.h5'
CLASS_NAMES_PATH = 'class_names.pkl'

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'replace_this_with_a_secret')


# ========================
# Load Model
# ========================
def load_model_safe():
    if not os.path.exists(MODEL_ARCH_PATH) or not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError("Model files not found. Place architecture and weights in the app root.")

    with open(MODEL_ARCH_PATH, 'rb') as f:
        model_json = pickle.load(f)

    model = model_from_json(model_json)
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model


# ========================
# Load Class Names
# ========================
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'rb') as f:
            try:
                class_names = pickle.load(f)
                return class_names
            except Exception:
                return None
    return None


# ========================
# Preprocessing
# ========================
def preprocess_image(image: Image.Image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    image = image.convert('RGB')
    image = image.resize(target_size)
    arr = np.array(image).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ========================
# Initialize Model
# ========================
try:
    model = load_model_safe()
    class_names = load_class_names()
    if class_names is None:
        num_classes = model.output_shape[-1]
        class_names = [f'class_{i}' for i in range(num_classes)]
        print("⚠️ class_names.pkl not found — using generic class_i labels.")
    else:
        print(f"✅ Loaded {len(class_names)} class names.")
except Exception as e:
    model = None
    class_names = None
    print("❌ ERROR loading model on startup:", str(e))


# ========================
# Flask Routes
# ========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash("Model not loaded. Check server logs.")
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash("No file part.")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash("No selected file.")
        return redirect(url_for('index'))

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        x = preprocess_image(image)
        preds = model.predict(x)
        top_idx = int(np.argmax(preds, axis=-1)[0])
        confidence = float(np.max(preds))
        label = class_names[top_idx] if class_names else f'class_{top_idx}'

        return render_template('index.html', filename=file.filename, label=label, confidence=f'{confidence:.4f}')

    except Exception as e:
        flash("Prediction error: " + str(e))
        return redirect(url_for('index'))


# ========================
# Run App
# ========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

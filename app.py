import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

# Load the model in the SavedModel format
model_path = '/content/drive/MyDrive/Model/EfficientNetV2B0/H5/EfficientNetV2B0.h5'
try:
    import os
    import tensorflow as tf

# Define the directory where the model is stored
    model_dir = "model"

# Define the filename of the model
    model_filename = "EfficientNetV2B0.h5"

# Construct the full path to the model file by joining the directory and filename
    model_path = os.path.join(model_dir, model_filename)

# Load the TensorFlow Keras model from the specified file path
    model = tf.keras.models.load_model(model_path)

    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Class names
class_names = ['Asoka', 'Bunga Telang', 'Daun Jambu Biji', 'Daun Jarak', 'Daun Jeruk Nipis', 'Daun Pepaya', 'Kayu Putih', 'Lidah Buaya', 'Semanggi', 'Sirih']

def load_and_process_image(image_data, target_size=(224, 224)):
    img = image.load_img(image_data, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Image data not found'}), 400

        image_file = request.files['image']
        image_data = io.BytesIO(image_file.read())
        img_array = load_and_process_image(image_data)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = tf.argmax(predictions[0]).numpy()
        confidence = predictions[0][predicted_class]

        # Return prediction result
        return jsonify({
            'predicted_class': class_names[predicted_class],
            'confidence': float(confidence)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

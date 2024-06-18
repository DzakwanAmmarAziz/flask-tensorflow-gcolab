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
try:
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

# Class names and plant information
class_names = ['Asoka', 'Bunga Telang', 'Daun Jambu Biji', 'Daun Jarak', 'Daun Jeruk Nipis', 'Daun Pepaya', 'Kayu Putih', 'Lidah Buaya', 'Semanggi', 'Sirih']
plant_info = {
    'Asoka': {
        'binomial': "Saraca Asoca",
        'description': "Tumbuhan ini dikenal sebagai simbol keindahan dan kedamaian dalam banyak budaya di seluruh dunia...",
        'benefit': [
            "1. Meringankan nyeri haid",
            "2. Menjaga kesehatan kulit",
            "3. Obat anti radang"
        ]
    },
    'Bunga Telang': {
        'binomial': "Clitoria Ternatea",
        'description': "Bunga Telang atau Clitoria ternatea, umumnya dikenal dengan 'butterfly pea'...",
        'benefit': [
            "1. Menurunkan demam dan meredakan rasa nyeri",
            "2. Meredakan gejala alergi",
            "3. Melancarkan aliran darah ke kapiler mata"
        ]
    },
    'Daun Jambu Biji': {
        'binomial': "Psidium guajava",
        'description': "Daun jambu biji (Psidium guajava) adalah bagian dari pohon jambu biji yang tumbuh di daerah tropis...",
        'benefit': [
            "1. Meningkatkan kekebalan tubuh",
            "2. Menjaga kadar gula darah tetap stabil",
            "3. Menjaga kesehatan sistem pencernaan"
        ]
    },
    'Daun Jarak': {
        'binomial': "Ricinus Communis",
        'description': "Daun jarak (Jatropha curcas) adalah tanaman herbal yang tumbuh subur di daerah tropis...",
        'benefit': [
            "1. Mengatasi sembelit",
            "2. Mengunci kelembapan kulit",
            "3. Mempercepat penyembuhan luka"
        ]
    },
    'Daun Jeruk Nipis': {
        'binomial': "Citrus Aurantifolia",
        'description': "Daun jeruk nipis (Citrus aurantiifolia) adalah tanaman herbal yang sering digunakan dalam berbagai pengobatan tradisional...",
        'benefit': [
            "1. Mempercepat penyembuhan luka...",
            "2. Meningkatkan kesehatan kulit...",
            "3. Mengatasi masalah pencernaan..."
        ]
    },
    'Daun Pepaya': {
        'binomial': "Carica Pepaya",
        'description': "Daun pepaya (Carica papaya) adalah bagian dari tanaman pepaya yang memiliki berbagai manfaat kesehatan...",
        'benefit': [
            "1. Meningkatkan pencernaan...",
            "2. Anti-inflamasi...",
            "3. Menurunkan tekanan darah..."
        ]
    },
    'Kayu Putih': {
        'binomial': "Melaleuca Leucadendra",
        'description': "Daun kayu putih (Melaleuca alternifolia) merupakan bagian dari pohon kayu putih yang tumbuh di Australia...",
        'benefit': [
            "1. Pereda sakit...",
            "2. Respon imun...",
            "3. Kondisi pernafasan..."
        ]
    },
    'Lidah Buaya': {
        'binomial': "Aloe Vera",
        'description': "Lidah buaya (Aloe vera), adalah spesies tanaman dengan daun berdaging tebal dari genus Aloe...",
        'benefit': [
            "1. Membantu mengatasi permasalahan kulit...",
            "2. Memelihara kesehatan kulit...",
            "3. Membantu menutrisi rambut..."
        ]
    },
    'Semanggi': {
        'binomial': "Marsilea Crenata Presl",
        'description': "Daun semanggi adalah bagian dari tanaman semanggi (Oxalis spp.) yang tumbuh secara alami di berbagai belahan dunia...",
        'benefit': [
            "1. Antioksidan yang kuat...",
            "2. Meningkatkan kekuatan tulang...",
            "3. Mendukung kesehatan prostat..."
        ]
    },
    'Sirih': {
        'binomial': "Piper betle",
        'description': "Daun sirih (Piper betle) adalah bagian dari tanaman sirih yang tumbuh subur di berbagai daerah tropis...",
        'benefit': [
            "1. Menyehatkan saluran pencernaan...",
            "2. Mengatasi sembelit...",
            "3. Menjaga kesehatan mulut dan gigi..."
        ]
    }
}

class_names = list(plant_info.keys())

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
        is_above_threshold = bool(confidence > 0.5)  # Convert to regular Python boolean

        # Get plant info
        predicted_plant = class_names[predicted_class]
        plant_details = plant_info[predicted_plant]

        # Return prediction result
        return jsonify({
            'message': 'Model is predicted successfully.',
            'data': {
                'result': predicted_plant,
                'confidenceScore': float(confidence * 100),  # Convert to percentage
                'isAboveThreshold': is_above_threshold,
                'binomial': plant_details['binomial'],
                'description': plant_details['description'],
                'benefit': plant_details['benefit']
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import keras.backend as K
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = r'C:\Users\calar\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None  # On initialise le modèle à l'extérieur de toute fonction


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def f2_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f2_val = 5 * (precision * recall) / (4 * precision + recall + K.epsilon())
    return f2_val


def load_model():
    global model
    # Chargement du modèle ici
    model_path = r"C:/Users/calar/OneDrive/Bureau/Etudes/BoneIdea/boneidea_be/fracturejuly_real.h5"
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_score': f1_score, 'f2_score': f2_score})



def preprocess_image(img_path, img_dim=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image à {img_path}. Vérifiez le chemin d'accès et l'intégrité du fichier.")
    img = cv2.resize(img, img_dim)
    img = img / 255.0

    # Imprimer la forme et quelques valeurs de pixels pour vérification
    print(f"Forme de l'image après prétraitement: {img.shape}")
    print(f"Quelques valeurs de pixels: {img[0, 0]}, {img[100, 100]}, {img[223, 223]}")

    return img


# Fonction pour prédire une image spécifique
def predict_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(np.expand_dims(img, axis=0))
    return prediction[0][0]
  


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():
    logging.info('Prediction request received')
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            pred_value = predict_image(file_path)  # Utilisez la fonction predict_image ici
        except Exception as e:
            return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

        label = "FRACTURE" if pred_value > 0.5 else "NO FRACTURE"
        probability = pred_value if label == "FRACTURE" else (1 - pred_value)
        
        #tf.keras.backend.clear_session()

        return jsonify({"filename": filename, "prediction": label, "probability": float(probability)})
    if os.path.exists(file_path):
            os.remove(file_path)
    
    return jsonify({"error": "Invalid file type"}), 400


if __name__ == "__main__":
    load_model()  # Charger le modèle au démarrage de l'application

    
    print("Test sur une image avec chemin précisé")
    test_chemin = r"C:\Users\calar\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\train\not fractured\10-rotated1-rotated1-rotated1-rotated1.jpg"
    predictionn = predict_image(test_chemin)
    if predictionn > 0.5:
        print(f"Le modèle prédit que l'image {test_chemin} est une fracture")
    else: 
        print(f"Le modèle prédit que l'image {test_chemin} n'est pas une fracture")


    # Testez le modèle sur une image
    print("test sur 2 images")
    
    fractured_dir = r'C:\Users\calar\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\train\fractured'
    not_fractured_dir = r'C:\Users\calar\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\train\not fractured'

    not_fractured_files = [os.path.join(not_fractured_dir, f) for f in os.listdir(not_fractured_dir)[:5]]

    for image_path in not_fractured_files:
        prediction2 = predict_image(image_path)
        if prediction2 > 0.5:
            print(f"Le deuxieme modèle prédit que l'image {image_path} EST une fracture avec une confiance de {(prediction2)*100:.2f}%.")
        else:
            print(f"Le deuxieme modèle prédit que l'image {image_path} N'EST PAS une fracture avec une confiance de {(1-prediction2)*100:.2f}%.")


    print("Le serveur est lancé et en cours d'écoute")
    app.run(host='0.0.0.0', port=5000, debug=True)

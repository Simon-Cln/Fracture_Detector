from keras.models import load_model
import cv2
import numpy as np


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

# Charger le modèle
model = load_model("C:/Users/calar/OneDrive/Bureau/Etudes/BoneIdea/boneidea_be/fracturejuly_real.h5",custom_objects={'f1_score': f1_score, 'f2_score': f2_score})
# Fonction pour prétraiter les images
def preprocess_image(img_path, img_dim=(224, 224)):
    img = cv2.imread(img_path)
    
    if img is None:
        raise ValueError(f"Unable to read image at {img_path}. Check the file path and integrity.")
    
    img = cv2.resize(img, img_dim)
    img = img / 255.0
    return img


# Fonction pour prédire une image spécifique
def predict_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(np.expand_dims(img, axis=0))
    return prediction[0][0]

import os

# Chemins vers les dossiers fractured et not fractured
fractured_dir = r'C:\Users\calar\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\train\fractured'
not_fractured_dir = r'C:\Users\calar\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\train\not fractured'

# Obtenir les 20 premiers fichiers des deux dossiers
fractured_files = [os.path.join(fractured_dir, f) for f in os.listdir(fractured_dir)[:20:]]
not_fractured_files = [os.path.join(not_fractured_dir, f) for f in os.listdir(not_fractured_dir)[:20:]]

# Fusionner les deux listes
all_files = fractured_files + not_fractured_files

# Prédire pour chaque image
for img_path in all_files:
    prediction = predict_image(img_path)
    if prediction > 0.5:
        print(f"Le modèle prédit que l'image {img_path} est une FRACTURE avec une confiance de {prediction*100:.2f}%.")
    else:
        print(f"Le modèle prédit que l'image {img_path} N'EST PAS une fracture avec une confiance de {(1-prediction)*100:.2f}%.")

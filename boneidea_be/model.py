import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.model_selection import train_test_split

# Charger les images
def load_images_from_directory(directory, label, img_dim=(224, 224), max_images=None):
    X, y = [], []
    image_count = 0
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if max_images and image_count >= max_images:
                break
            if filename[-3:] == "jpg":
                img_path = os.path.join(dirname, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_dim)
                img = img / 255.0
                X.append(img)
                y.append(label)
                image_count += 1
    return X, y

# Dossiers
train_dir = r'C:\Users\calar\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\train'
val_dir = r'C:\Users\calar\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\val'

# Charger les images naturelles et des fractures osseuses
max_images_per_category = 700  # Pour un total de 500 images (250 pour chaque catégorie)
train_natural_dir = os.path.join(train_dir, 'not fractured')
train_fractures_dir = os.path.join(train_dir, 'fractured')

val_natural_dir = os.path.join(val_dir, 'not fractured')
val_fractures_dir = os.path.join(val_dir, 'fractured')

X_train_natural, y_train_natural = load_images_from_directory(train_natural_dir, 0, max_images=max_images_per_category)
X_train_fractures, y_train_fractures = load_images_from_directory(train_fractures_dir, 1, max_images=max_images_per_category)

X_val_natural, y_val_natural = load_images_from_directory(val_natural_dir, 0, max_images=max_images_per_category)
X_val_fractures, y_val_fractures = load_images_from_directory(val_fractures_dir, 1, max_images=max_images_per_category)

# Fusionner les listes d'entraînement
X_train = X_train_natural + X_train_fractures
y_train = y_train_natural + y_train_fractures

# Fusionner les listes de validation
X_val = X_val_natural + X_val_fractures
y_val = y_val_natural + y_val_fractures

# Convertir en np.array
X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = np.array(y_val)


# Séparation en ensembles d'entraînement et de validation


# Architecture du modèle
input_shape = (224, 224, 3)
model = Sequential([
    Conv2D(32, (2, 2), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (2, 2), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

from keras import backend as K

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

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy', f1_score, f2_score])

# Formation
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Enregistrer le modèle
model.save("fracturejuly_real.h5")

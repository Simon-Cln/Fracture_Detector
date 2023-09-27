import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from keras.regularizers import l1_l2

# define paths to your train and validation directories
train_dir = r'C:\Users\Simon\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\train'
val_dir = r'C:\Users\Simon\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\val'

# define the constants
IMG_SIZE = (224, 224)  # taille des images à laquelle VGG16 s'attend
BATCH_SIZE = 32

# create a data generator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Increase the rotation range for data augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'  # Add this to fill in pixels after a rotation or shift
)

val_datagen = ImageDataGenerator(rescale=1./255)  # validation data shouldn't be augmented

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load the model
model = load_model('InceptionV3_model.h5')

# Check model summary
model.summary()

# Add regularization
for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = l1_l2(l1=1e-5, l2=1e-4)

# Create callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

# Continue training with more epochs and smaller learning rate
opt = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=50,  # Increase the number of epochs
    callbacks=[reduce_lr]
)

# Save the model
model.save('InceptionV3_model_continued.h5')

# Save the history
with open('InceptionV3_history_continued.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Model Loss for InceptionV3 Continued Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Model Accuracy for InceptionV3 Continued Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(val_generator)

# Convert probabilities to class labels
y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred]

# Get true class labels
y_true = val_generator.classes

# Calculate F2 score
f2 = fbeta_score(y_true, y_pred, beta=2)
print(f'F2 score: {f2}')

# Calculate precision
precision = precision_score(y_true, y_pred)
print(f'Precision: {precision}')

# Calculate recall
recall = recall_score(y_true, y_pred)
print(f'Recall: {recall}')

# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f'F1 score: {f1}')

# Calculate AUC-ROC
auc_roc = roc_auc_score(y_true, y_pred)
print(f'AUC-ROC: {auc_roc}')

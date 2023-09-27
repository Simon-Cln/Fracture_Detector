import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50, InceptionV3
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# define paths to your train and validation directories
train_dir = r'C:\Users\Simon\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\train'
val_dir = r'C:\Users\Simon\OneDrive\Bureau\Etudes\BoneIdea\boneidea_be\val'

# define the constants
IMG_SIZE = (224, 224)  # taille des images à laquelle VGG16 s'attend
BATCH_SIZE = 32

# create a data generator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)  # validation data shouldn't be augmented

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'  # as we have binary classification problem
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Initialize a basic CNN model
def create_cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

models = []

# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel1 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
baseModel2 = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
baseModel3 = InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# construct the head of the model that will be placed on top of the base model
headModel1 = baseModel1.output
headModel1 = Flatten(name="flatten")(headModel1)
headModel1 = Dense(512, activation="relu")(headModel1)
headModel1 = Dropout(0.5)(headModel1)
headModel1 = Dense(1, activation="sigmoid")(headModel1)
# For ResNet50
headModel2 = baseModel2.output
headModel2 = Flatten(name="flatten")(headModel2)
headModel2 = Dense(512, activation="relu")(headModel2)
headModel2 = Dropout(0.5)(headModel2)
headModel2 = Dense(1, activation="sigmoid")(headModel2)
# For InceptionV3
headModel3 = baseModel3.output
headModel3 = Flatten(name="flatten")(headModel3)
headModel3 = Dense(512, activation="relu")(headModel3)
headModel3 = Dropout(0.5)(headModel3)
headModel3 = Dense(1, activation="sigmoid")(headModel3)

# place the head FC model on top of the base model (this will become the actual model we will train)
model1 = Model(inputs=baseModel1.input, outputs=headModel1)
# Place the head FC model on top of the base model (this will become the actual model we will train)
model2 = Model(inputs=baseModel2.input, outputs=headModel2)
model3 = Model(inputs=baseModel3.input, outputs=headModel3)

# loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in baseModel1.layers:
    layer.trainable = False

# Loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in baseModel2.layers:
    layer.trainable = False

for layer in baseModel3.layers:
    layer.trainable = False



models.append(("VGG16", model1))
models.append(("ResNet50", model2))
models.append(("InceptionV3", model3))
models.append(("BasicCNN", create_cnn((224, 224, 3))))

history = []

import time


# loop over all models
for (name, model) in models:
    print(f"Training model: {name}")

    # compile our model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Create callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

    start_time = time.time()

    # train the head of the network
    H = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=20,  # you can change the number of epochs
        callbacks=[early_stop, reduce_lr]
    )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time for model {name}: {total_time} seconds")
    print(f"Training for model {name} completed.")

    # save the model
    model.save(f"{name}_model.h5")

    # save the history
    with open(f"{name}_history.pkl", 'wb') as f:
        pickle.dump(H.history, f)

    # Append the history to compare later
    history.append((name, H))

# Plotting performance
print("Plotting performance...")

# loop over the history
for (model_name, H) in history:
    # plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(H.history['loss'], label='train')
    plt.plot(H.history['val_loss'], label='val')
    plt.title('Model Loss for {}'.format(model_name))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(H.history['accuracy'], label='train')
    plt.plot(H.history['val_accuracy'], label='val')
    plt.title('Model Accuracy for {}'.format(model_name))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Print the best model
best_model = max(history, key=lambda item:item[1].history['val_accuracy'][-1])
print(f"The best model is: {best_model[0]} with a validation accuracy of {best_model[1].history['val_accuracy'][-1]*100:.2f}%")

# Provide further recommendation based on the best model's performance
if best_model[1].history['val_accuracy'][-1] >= 0.95:
    print("This model has achieved a high accuracy and can be used for predictions.")
else:
    print("This model could benefit from further training or hyperparameter tuning.")

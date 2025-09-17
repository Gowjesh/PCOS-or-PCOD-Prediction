# PCOS CLASSIFIER - FULL SCRIPT (SAFE VERSION)
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. FOLDER PATHS
train_dir = "dataset/train"  # Must contain 'Normal' and 'PCOS' subfolders
test_dir = "dataset/test_images"  # Folder containing new images for prediction
model_path = "pcos_model.h5"

# 2. CLEAN CORRUPTED IMAGES
def clean_folder(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                img = Image.open(path)
                img.verify()  # Check if the image is corrupted
            except (IOError, UnidentifiedImageError, SyntaxError):
                print(f"Removing corrupted image: {path}")
                os.remove(path)

# Clean both training subfolders
clean_folder(os.path.join(train_dir, "Normal"))
clean_folder(os.path.join(train_dir, "PCOS"))

# 3. IMAGE DATA GENERATOR
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=16,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=16,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# 4. BUILD CNN MODEL
def create_cnn(input_shape=(128,128,3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_cnn()
model.summary()

# 5. TRAIN MODEL
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25
)

# Save trained model
model.save(model_path)
print(f"Model saved as {model_path}")

# 6. PLOT ACCURACY & LOSS
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy over Epochs")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss over Epochs")
plt.show()

# 7. FUNCTION TO PREDICT IMAGE
def predict_image(model, img_path):
    try:
        img = Image.open(img_path).convert('RGB').resize((128,128))
        img_array = np.expand_dims(np.array(img)/255.0, axis=0)
        pred = model.predict(img_array)
        label = "PCOS" if pred[0][0] > 0.5 else "Normal"

        plt.imshow(img)
        plt.title(f"Prediction: {label}")
        plt.axis('off')
        plt.show()
        return label
    except UnidentifiedImageError:
        print(f"Cannot open image: {img_path}")
        return None

# 8. PREDICT NEW IMAGES
print("Predicting images in folder:", test_dir)
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    label = predict_image(model, img_path)
    if label:
        print(f"{img_file} → {label}")

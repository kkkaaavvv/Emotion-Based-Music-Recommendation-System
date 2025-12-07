# train_cnn_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Data augmentation + preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
)

# Build CNN model
model = Sequential(
    [
        Conv2D(64, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(7, activation="softmax"),  # 7 emotion classes
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Save best model
os.makedirs("trained_model", exist_ok=True)

checkpoint = ModelCheckpoint(
    "trained_model/model.h5", monitor="val_accuracy", verbose=1, save_best_only=True
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)

# Train model
history = model.fit(
    train_data, validation_data=test_data, epochs=40, callbacks=[checkpoint, early_stop]
)

print("Training complete! Model saved to trained_model/model.h5")

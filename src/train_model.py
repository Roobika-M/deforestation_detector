# src/train_model.py

import rasterio
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import cv2

# --- Configuration ---
# Define your file paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'deforestation_data')
input_path = os.path.join(DATA_FOLDER, 'deforestation_input_image.tif')
labels_path = os.path.join(DATA_FOLDER, 'deforestation_labels.tif')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'deforestation_3band_model.h5')
IMAGE_SIZE = (256, 256)

# --- 1. Load Data, Pad, and Create Tiles ---
print("Loading data...")
try:
    with rasterio.open(input_path) as src:
        full_input_image = src.read()

    with rasterio.open(labels_path) as src:
        full_labels = src.read(1)

    print("Original input image shape:", full_input_image.shape)
    print("Original labels shape:", full_labels.shape)
except FileNotFoundError:
    print(f"Error: Data files not found. Please ensure '{DATA_FOLDER}' folder exists with the following files:")
    print(f"- {input_path}")
    print(f"- {labels_path}")
    exit()

# Transpose image to H, W, C format for TensorFlow
full_input_image = np.transpose(full_input_image, (1, 2, 0))

# Normalize the input image data
full_input_image = full_input_image.astype('float32') / 4000.0
# Ensure labels are binary (0 or 1)
full_labels = (full_labels > 0).astype(np.float32)

# The UNet model expects squared images, so let's resize them
resized_input = cv2.resize(full_input_image, IMAGE_SIZE)
resized_labels = cv2.resize(full_labels, IMAGE_SIZE)
resized_labels = np.expand_dims(resized_labels, axis=-1)

# Add a batch dimension
X = np.expand_dims(resized_input, axis=0)
y = np.expand_dims(resized_labels, axis=0)

# --- 2. Define the UNet Model Architecture ---
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = Conv2D(64, 2, activation='relu', padding='same')(up4)
    merge4 = concatenate([conv2, up4], axis=3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(merge4)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(32, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv1, up5], axis=3)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(conv5)
    
    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=output)
    return model

# --- 3. Train the Model ---
print("\n--- Step 3: Training the model ---")
model = unet_model(input_size=(256, 256, 3))
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model with the preprocessed data
# Note: You need a larger dataset for effective training
model.fit(x=X, y=y, epochs=10, batch_size=1)

# --- 4. Save the Model ---
print("\n--- Step 4: Saving the model ---")
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

model.save(MODEL_PATH)
print(f"Model saved successfully to {MODEL_PATH}")
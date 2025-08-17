# detect_deforestation.py

import rasterio
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_PATH = 'deforestation_3band_model.h5'
LIVE_IMAGE_FOLDER = 'deforestation_alerts'
LIVE_IMAGE_PATH = os.path.join(LIVE_IMAGE_FOLDER, 'latest_satellite_image.tif')
TILE_SIZE = 256

# --- 1. Load Model and Latest Satellite Image ---
print("Loading the trained model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'deforestation_3band_model.h5' exists in your project folder.")
    exit()

print(f"Loading the latest satellite image from: {LIVE_IMAGE_PATH}")
try:
    with rasterio.open(LIVE_IMAGE_PATH) as src:
        live_image = src.read()
except FileNotFoundError:
    print(f"Error: The image file '{LIVE_IMAGE_PATH}' was not found.")
    print("Please run fetch_data.py to download a new image first.")
    exit()

# --- 2. Preprocess the Image and Create Tiles ---
def preprocess_and_tile_image(image_data):
    # Normalize the image data
    image_data = image_data.astype('float32') / 3000.0
    # Transpose to (height, width, channels)
    image_data = np.transpose(image_data, (1, 2, 0))

    height, width, _ = image_data.shape
    padded_height = (height // TILE_SIZE + 1) * TILE_SIZE
    padded_width = (width // TILE_SIZE + 1) * TILE_SIZE
    
    padded_image = np.pad(image_data, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode='constant')

    tiled_images = []
    for y in range(0, padded_height, TILE_SIZE):
        for x in range(0, padded_width, TILE_SIZE):
            tile = padded_image[y:y + TILE_SIZE, x:x + TILE_SIZE, :]
            tiled_images.append(tile)
    
    return np.array(tiled_images), height, width

processed_image_tiles, original_height, original_width = preprocess_and_tile_image(live_image)

# --- 3. Make Prediction with the Model on each tile ---
print("Making predictions on image tiles...")
predictions = model.predict(processed_image_tiles)

# --- 4. Stitch predictions back together ---
stitched_mask = np.zeros((processed_image_tiles.shape[0] * TILE_SIZE, TILE_SIZE, 1), dtype=np.float32)

num_tiles_x = (original_width // TILE_SIZE + 1)
num_tiles_y = (original_height // TILE_SIZE + 1)

stitched_mask = np.zeros((num_tiles_y * TILE_SIZE, num_tiles_x * TILE_SIZE, 1), dtype=np.float32)

tile_index = 0
for y in range(num_tiles_y):
    for x in range(num_tiles_x):
        stitched_mask[y * TILE_SIZE : (y + 1) * TILE_SIZE, x * TILE_SIZE : (x + 1) * TILE_SIZE, :] = predictions[tile_index]
        tile_index += 1

# Crop the stitched mask to the original image size
final_mask = stitched_mask[:original_height, :original_width, :]

# Convert prediction to a binary mask (0 or 1)
final_prediction_mask = (final_mask > 0.5).astype(np.uint8)

# --- 5. Check for Deforestation and Provide an Alert ---
# Calculate the percentage of predicted deforestation pixels
deforestation_percentage = (np.sum(final_prediction_mask) / final_prediction_mask.size) * 100

print(f"Deforestation detected: {deforestation_percentage:.2f}%")

if deforestation_percentage > 0.1: # You can adjust this threshold
    print("\n🚨🚨🚨 ALERT! Deforestation Detected! 🚨🚨🚨")
    print(f"Deforestation was detected with a change of {deforestation_percentage:.2f}%.")
else:
    print("\n✅ No significant deforestation detected.")

# --- 6. Visualize the Results ---
# Load original image for visualization
with rasterio.open(LIVE_IMAGE_PATH) as src:
    original_image_rgb = (np.transpose(src.read(), (1, 2, 0)) / src.read().max() * 255).astype(np.uint8)
    
deforestation_overlay = original_image_rgb.copy()
# Change color to red for deforestation
deforestation_overlay[final_prediction_mask[:, :, 0] == 1] = [255, 0, 0]

# Blend the original image with the overlay
blended_image = Image.fromarray(original_image_rgb).convert("RGBA")
red_mask = Image.fromarray(deforestation_overlay).convert("RGBA")
blended_image = Image.blend(blended_image, red_mask, alpha=0.5)

# Display the results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(original_image_rgb)
axes[0].set_title('Original Satellite Image')
axes[0].axis('off')

axes[1].imshow(final_prediction_mask[:, :, 0], cmap='gray')
axes[1].set_title('Predicted Deforestation Mask')
axes[1].axis('off')

axes[2].imshow(blended_image)
axes[2].set_title('Overlay with Deforestation Alert')
axes[2].axis('off')

plt.show()
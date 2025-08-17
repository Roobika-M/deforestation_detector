# src/deforestation_engine.py

import rasterio
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import io

# --- Configuration ---
TILE_SIZE = 256

def run_detection_on_image(image_path, model_path='deforestation_3band_model.h5'):
    """
    Runs deforestation detection on a given image file.

    Args:
        image_path (str): The file path to the satellite image.
        model_path (str): The file path to the trained model.

    Returns:
        tuple: A tuple containing the blended image, the mask image,
               and the percentage of deforestation.
    """
    # Load the trained model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

    # Load the latest satellite image
    try:
        with rasterio.open(image_path) as src:
            live_image = src.read()
    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
        return None, None, None

    # --- Preprocess the Image and Create Tiles ---
    def preprocess_and_tile_image(image_data):
        image_data = image_data.astype('float32') / 3000.0
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

    # --- Make Prediction with the Model on each tile ---
    print("Making predictions on image tiles...")
    predictions = model.predict(processed_image_tiles)

    # --- Stitch predictions back together ---
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
    final_prediction_mask = (final_mask > 0.5).astype(np.uint8)

    # --- Check for Deforestation ---
    deforestation_percentage = (np.sum(final_prediction_mask) / final_prediction_mask.size) * 100

    # --- Visualize the Results ---
    with rasterio.open(image_path) as src:
        original_image_rgb = (np.transpose(src.read(), (1, 2, 0)) / src.read().max() * 255).astype(np.uint8)
        
    deforestation_overlay = original_image_rgb.copy()
    deforestation_overlay[final_prediction_mask[:, :, 0] == 1] = [255, 0, 0]

    blended_image = Image.fromarray(original_image_rgb).convert("RGBA")
    red_mask = Image.fromarray(deforestation_overlay).convert("RGBA")
    blended_image = Image.blend(blended_image, red_mask, alpha=0.5)

    return blended_image, Image.fromarray(final_prediction_mask[:, :, 0] * 255), deforestation_percentage
# src/deforestation_engine.py

import os
import tensorflow as tf
import numpy as np
import rasterio
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global model and tile size
MODEL = None
TILE_SIZE = 256
THRESHOLD = 0.5   # A global threshold for the model's output
DISPLAY_SIZE = (640, 480)   # A smaller fixed size for the output images

def load_and_preprocess_image(image_path):
    """
    Loads, preprocesses, and tiles a satellite image for model inference.

    Args:
        image_path (str): The absolute path to the input TIFF image.

    Returns:
        tuple: A tuple containing the list of image tiles, original image shape, and original image array.
               Returns (None, None, None) if an error occurs.
    """
    if not os.path.exists(image_path):
        logging.error(f"Error: Input image not found at {image_path}")
        return None, None, None

    try:
        with rasterio.open(image_path) as src:
            # Read all available bands from the image, not just the first 3
            image_array = src.read()
            original_shape = (src.height, src.width)
            
        # Transpose to HWC (Height, Width, Channels)
        image_array = np.transpose(image_array, (1, 2, 0))

        # Pad the image to be a multiple of the tile size
        pad_h = TILE_SIZE - (original_shape[0] % TILE_SIZE) if original_shape[0] % TILE_SIZE != 0 else 0
        pad_w = TILE_SIZE - (original_shape[1] % TILE_SIZE) if original_shape[1] % TILE_SIZE != 0 else 0
        padded_image = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')

        # Normalize the image data. Add a small epsilon to avoid division by zero.
        max_val = np.max(padded_image)
        if max_val == 0:
            logging.warning("Image max value is 0. Returning empty data.")
            return None, None, None
            
        padded_image = padded_image.astype('float32') / max_val
        
        # Create tiles
        tiles = []
        for y in range(0, padded_image.shape[0], TILE_SIZE):
            for x in range(0, padded_image.shape[1], TILE_SIZE):
                tile = padded_image[y:y+TILE_SIZE, x:x+TILE_SIZE, :]
                tiles.append(tile)

        return np.array(tiles), original_shape, image_array

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None, None, None


def run_detection_on_comparison(initial_image_path, final_image_path, model_path):
    """
    Loads two images, runs a pre-trained model on each, and compares the results
    to detect new areas of deforestation.

    Args:
        initial_image_path (str): Path to the "before" image.
        final_image_path (str): Path to the "after" image.
        model_path (str): The absolute path to the trained Keras model file.

    Returns:
        tuple: A tuple containing the blended image (PIL.Image), the mask image
               (PIL.Image), and the percentage of new deforestation. Returns (None, None, 0)
               if an error occurs.
    """
    global MODEL
    
    if MODEL is None:
        if not os.path.exists(model_path):
            logging.error(f"Error: Model not found at {model_path}")
            return None, None, 0
        try:
            logging.info("Loading deforestation detection model...")
            # Remember to update this path to your new 4-band model file after training
            MODEL = load_model(model_path, compile=False)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None, None, 0

    logging.info("Processing initial image...")
    initial_tiles, initial_shape, initial_image_array = load_and_preprocess_image(initial_image_path)
    if initial_tiles is None:
        return None, None, 0

    logging.info("Processing final image...")
    final_tiles, final_shape, final_image_array = load_and_preprocess_image(final_image_path)
    if final_tiles is None:
        return None, None, 0

    # Ensure images have the same dimensions for comparison
    if initial_shape != final_shape:
        logging.warning("Images have different dimensions. This may cause issues.")

    logging.info("Running detection on initial image...")
    initial_predictions = MODEL.predict(initial_tiles, verbose=0)
    # The mask is now correctly a binary array (0s and 1s)
    initial_mask_binary = postprocess_predictions(initial_predictions, initial_shape)

    logging.info("Running detection on final image...")
    final_predictions = MODEL.predict(final_tiles, verbose=0)
    # The mask is now correctly a binary array (0s and 1s)
    final_mask_binary = postprocess_predictions(final_predictions, final_shape)

    # --- Compare the two masks to find new deforestation ---
    # Find pixels that are deforested in the final image but not in the initial image.
    new_deforestation_mask = np.logical_and(final_mask_binary, np.logical_not(initial_mask_binary))

    # --- Calculate percentage ---
    # We're calculating the percentage of the total image area that has experienced *new* deforestation.
    deforestation_pixels = np.sum(new_deforestation_mask)
    total_pixels = new_deforestation_mask.size
    deforestation_percentage = (deforestation_pixels / total_pixels) * 100

    # --- Create visual output ---
    # Create a red mask for the newly detected areas
    final_image_pil = Image.fromarray(final_image_array.astype(np.uint8))
    
    # Rescale the new_deforestation_mask for visual blending (0 to 255)
    new_deforestation_visual = (new_deforestation_mask.astype(np.uint8) * 255)
    
    red_mask_array = np.zeros((*final_shape, 4), dtype=np.uint8)
    red_mask_array[..., 0] = new_deforestation_visual   # Red channel
    red_mask_array[..., 3] = new_deforestation_visual * 0.7   # Alpha for transparency
    red_mask_pil = Image.fromarray(red_mask_array, 'RGBA')

    # Create a blended image by overlaying the red mask
    blended_image = Image.alpha_composite(final_image_pil.convert('RGBA'), red_mask_pil)
    
    # Create a simple black and white mask image from the new deforestation mask
    bw_mask = Image.fromarray(new_deforestation_visual, 'L')

    # Resize the final images for display
    blended_image = blended_image.resize(DISPLAY_SIZE, Image.LANCZOS)
    bw_mask = bw_mask.resize(DISPLAY_SIZE, Image.NEAREST)

    return blended_image, bw_mask, deforestation_percentage


def postprocess_predictions(predictions, original_shape):
    """
    Stitches prediction tiles back together into a single mask and resizes it to the original shape.
    This version returns a binary mask (0s and 1s) to avoid issues with float values.

    Args:
        predictions (np.array): The array of model predictions for each tile.
        original_shape (tuple): The (height, width) of the original image.

    Returns:
        np.array: The final, resized binary prediction mask (0s and 1s).
    """
    # Number of tiles in height and width
    num_tiles = predictions.shape[0]
    num_tiles_h = int(np.ceil(original_shape[0] / TILE_SIZE))
    num_tiles_w = int(np.ceil(original_shape[1] / TILE_SIZE))

    if num_tiles != num_tiles_h * num_tiles_w:
        logging.error("Mismatch in number of tiles. Prediction stitching may fail.")
        return None

    # Stitch tiles back together
    rows = []
    for i in range(num_tiles_h):
        row = predictions[i * num_tiles_w : (i + 1) * num_tiles_w]
        rows.append(np.hstack(row))
    stitched_mask = np.vstack(rows)

    # Convert predictions to a binary mask based on a threshold
    binary_mask = (stitched_mask[:, :, 0] > THRESHOLD)

    # Resize the binary mask to the original image dimensions
    mask_pil = Image.fromarray(binary_mask.astype(np.uint8) * 255, 'L')
    mask_resized_to_original = mask_pil.resize((original_shape[1], original_shape[0]), Image.NEAREST)
    mask_resized_array = np.array(mask_resized_to_original)
    
    # Convert the 0-255 array back to a 0-1 binary array for calculations
    return (mask_resized_array > 0).astype(np.uint8)

# Helper function for live alerts, using a single image
def run_detection_on_image(image_path, model_path):
    """
    Loads an image and runs a pre-trained model to detect deforestation.
    
    This function is kept for the "Live Alerts" section of the dashboard.
    
    Args:
        image_path (str): The absolute path to the input TIFF image.
        model_path (str): The absolute path to the trained Keras model file.

    Returns:
        tuple: A tuple containing the blended image (PIL.Image), the mask image
               (PIL.Image), and the percentage of deforestation. Returns (None, None, 0)
               if an error occurs.
    """
    global MODEL
    
    if MODEL is None:
        if not os.path.exists(model_path):
            logging.error(f"Error: Model not found at {model_path}")
            return None, None, 0
        try:
            logging.info("Loading deforestation detection model...")
            # Remember to update this path to your new 4-band model file after training
            MODEL = load_model(model_path, compile=False)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None, None, 0
    
    tiles, original_shape, image_array = load_and_preprocess_image(image_path)
    if tiles is None:
        return None, None, 0
    
    logging.info("Running detection on image...")
    predictions = MODEL.predict(tiles, verbose=0)
    prediction_mask_binary = postprocess_predictions(predictions, original_shape)
    
    if prediction_mask_binary is None:
        return None, None, 0

    # Calculate deforestation percentage
    deforestation_pixels = np.sum(prediction_mask_binary)
    total_pixels = prediction_mask_binary.size
    deforestation_percentage = (deforestation_pixels / total_pixels) * 100
    
    # Create visual output
    original_image_pil = Image.fromarray(image_array.astype(np.uint8))
    
    # Rescale the mask for visual blending (0 to 255)
    prediction_mask_visual = (prediction_mask_binary * 255).astype(np.uint8)
    
    red_mask_array = np.zeros((*original_shape, 4), dtype=np.uint8)
    red_mask_array[..., 0] = prediction_mask_visual   # Red channel
    red_mask_array[..., 3] = prediction_mask_visual * 0.7   # Alpha
    red_mask_pil = Image.fromarray(red_mask_array, 'RGBA')
    blended_image = Image.alpha_composite(original_image_pil.convert('RGBA'), red_mask_pil)
    
    # Create a simple black and white mask image from the resized mask
    bw_mask = Image.fromarray(prediction_mask_visual, 'L')
    
    # Resize the final images for display
    blended_image = blended_image.resize(DISPLAY_SIZE, Image.LANCZOS)
    bw_mask = bw_mask.resize(DISPLAY_SIZE, Image.NEAREST)

    return blended_image, bw_mask, deforestation_percentage
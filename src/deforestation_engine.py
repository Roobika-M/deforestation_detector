# src/deforestation_engine.py

import os
import tensorflow as tf
import numpy as np
import rasterio
from PIL import Image
import cv2

# Use a memory-efficient model when available
from tensorflow.keras.models import load_model

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Loads a GeoTIFF image, normalizes it, and resizes it.
    
    Args:
        image_path (str): The absolute path to the input TIFF image.
        target_size (tuple): The target dimensions for resizing.
        
    Returns:
        tuple: The original image dimensions and the preprocessed image array.
    """
    if not os.path.exists(image_path):
        print(f"Error: Preprocessing failed, image not found at {image_path}")
        return None, None
        
    try:
        with rasterio.open(image_path) as src:
            original_image_array = src.read()
            original_shape = (src.height, src.width)
            
        # Transpose from (channels, height, width) to (height, width, channels)
        original_image_array = np.transpose(original_image_array, (1, 2, 0))
        
        # Normalize the image data to a [0, 1] range for the model
        # Sentinel-2 images are 12-bit, so max value is around 4096, but it can go higher
        normalized_image = original_image_array.astype(np.float32) / 4000.0
        
        # Resize to the model's input size (e.g., 256x256)
        resized_image = cv2.resize(normalized_image, target_size, interpolation=cv2.INTER_AREA)
        
        # Add a batch dimension to the image
        resized_image_expanded = np.expand_dims(resized_image, axis=0)
        
        return original_shape, resized_image_expanded
        
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None, None

def run_detection_on_image(image_path, model_path):
    """
    Loads a new satellite image, processes it, and runs a pre-trained model
    to detect deforestation.

    Args:
        image_path (str): The absolute path to the input TIFF image.
        model_path (str): The absolute path to the trained Keras model file.

    Returns:
        tuple: A tuple containing the blended image (PIL.Image), the mask image
               (PIL.Image), and the percentage of deforestation. Returns (None, None, 0)
               if an error occurs.
    """
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}")
        return None, None, 0

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None, None, 0

    try:
        # Load the trained model
        print("Loading deforestation detection model...")
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
        
        # Preprocess the image
        original_shape, processed_image = preprocess_image(image_path, target_size=(256, 256))
        
        if original_shape is None:
            return None, None, 0
            
        # Make a prediction
        print("Making a prediction...")
        prediction = model.predict(processed_image)
        
        # Post-process the prediction to create a binary mask
        prediction_mask = (prediction > 0.5).astype(np.uint8) * 255
        
        # Resize the mask back to the original image's dimensions
        mask_pil = Image.fromarray(prediction_mask[0, :, :, 0], 'L')
        mask_resized_to_original = mask_pil.resize(original_shape[::-1], Image.LANCZOS)
        mask_resized_array = np.array(mask_resized_to_original)

        # Calculate deforestation percentage from the resized mask
        deforestation_pixels = np.sum(mask_resized_array > 0)
        total_pixels = mask_resized_array.size
        deforestation_percentage = (deforestation_pixels / total_pixels) * 100

        # Create visual output
        with rasterio.open(image_path) as src:
            original_image_array = np.transpose(src.read(), (1, 2, 0))

        # Normalize the original image for blending (if it's not already in 0-255 range)
        original_image_array = (original_image_array / original_image_array.max() * 255).astype(np.uint8)
        original_image_pil = Image.fromarray(original_image_array).convert('RGBA')
        
        # Create a red mask for the detected areas
        red_mask_array = np.zeros((*original_shape, 4), dtype=np.uint8)
        red_mask_array[..., 0] = mask_resized_array  # Red channel
        red_mask_array[..., 3] = mask_resized_array * 0.7  # Alpha for transparency
        red_mask_pil = Image.fromarray(red_mask_array, 'RGBA')

        # Create a blended image by overlaying the mask
        blended_image = Image.alpha_composite(original_image_pil, red_mask_pil)
        
        # Create a simple black and white mask image from the resized mask
        bw_mask = Image.fromarray(mask_resized_array, 'L')
        
        return blended_image, bw_mask, deforestation_percentage

    except Exception as e:
        print(f"An unexpected error occurred during detection: {e}")
        return None, None, 0
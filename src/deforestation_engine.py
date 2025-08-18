# src/deforestation_engine.py

import os
import tensorflow as tf
import numpy as np
import rasterio
from PIL import Image

# Use a memory-efficient model when available
from tensorflow.keras.models import load_model

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
        # The 'custom_objects' argument is important if the model uses custom layers
        print("Loading deforestation detection model...")
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
        
        # Get the required input shape from the model's first layer
        # The expected shape is (None, height, width, channels)
        input_shape = model.input_shape
        target_height = input_shape[1]
        target_width = input_shape[2]

        # Read the satellite image
        print(f"Reading satellite image from {image_path}...")
        with rasterio.open(image_path) as src:
            # We are expecting a 3-band RGB image from Sentinel-2
            image_array = src.read()
            original_shape = image_array.shape[1:]  # (height, width)
            image_array = np.moveaxis(image_array, 0, -1)
            print("Image read successfully.")

        # Preprocess the image for the model
        print("Preprocessing image...")
        
        # Resize the image to the model's expected input size
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        pil_image_resized = pil_image.resize((target_width, target_height), Image.LANCZOS)
        image_array_resized = np.array(pil_image_resized)
        
        # Normalize the pixel values to the range [0, 1]
        image_normalized = image_array_resized / 255.0
        # The model expects a batch dimension
        image_input = np.expand_dims(image_normalized, axis=0)

        # Make a prediction
        print("Running deforestation prediction...")
        prediction = model.predict(image_input)
        print("Prediction complete.")

        # Post-process the prediction
        # The output is a probability map
        prediction_mask = (prediction[0] > 0.5).astype(np.uint8)
        
        # Resize the mask back to the original image size for overlay
        mask_pil = Image.fromarray(prediction_mask[:, :, 0] * 255, 'L')
        mask_resized_to_original = mask_pil.resize(original_shape, Image.LANCZOS)
        mask_resized_array = np.array(mask_resized_to_original)

        # Calculate deforestation percentage from the resized mask
        deforestation_pixels = np.sum(mask_resized_array > 0)
        total_pixels = mask_resized_array.size
        deforestation_percentage = (deforestation_pixels / total_pixels) * 100

        # Create visual output
        # Convert original image to PIL for easy blending
        original_image_pil = Image.fromarray(image_array.astype(np.uint8))
        
        # Create a red mask for the detected areas
        red_mask_array = np.zeros((*original_shape, 4), dtype=np.uint8)
        red_mask_array[..., 0] = mask_resized_array  # Red channel
        red_mask_array[..., 3] = mask_resized_array * 0.7  # Alpha for transparency
        red_mask_pil = Image.fromarray(red_mask_array, 'RGBA')

        # Create a blended image by overlaying the mask
        blended_image = Image.alpha_composite(original_image_pil.convert('RGBA'), red_mask_pil)
        
        # Create a simple black and white mask image from the resized mask
        bw_mask = Image.fromarray(mask_resized_array, 'L').convert('RGB')

        return blended_image, bw_mask, deforestation_percentage

    except Exception as e:
        print(f"An unexpected error occurred during detection: {e}")
        return None, None, 0

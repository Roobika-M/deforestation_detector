# src/detect_deforestation.py

import os
import rasterio
import numpy as np
from PIL import Image
from deforestation_engine import run_detection_on_image
from fetch_data import fetch_live_image, authenticate_and_initialize_gee

# --- Configuration ---
# Use the same paths as defined in app.py for consistency
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIVE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'deforestation_alerts')
LIVE_IMAGE_PATH = os.path.join(LIVE_IMAGE_FOLDER, 'latest_satellite_image.tif')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'deforestation_3band_model.h5')

if __name__ == '__main__':
    print("Starting Deforestation Detection Process...")

    # --- 1. Authenticate with Google Earth Engine and Fetch Image ---
    print("\n--- Step 1: Fetching latest satellite image ---")
    
    # Example coordinates for a region in Bolivia
    lat, lon = -15.00, -62.99
    
    if not authenticate_and_initialize_gee():
        print("Failed to initialize Earth Engine. Exiting.")
        exit()

    if not fetch_live_image(lat, lon):
        print("Failed to download a new satellite image. Exiting.")
        exit()
    
    # Check if the downloaded file exists
    if not os.path.exists(LIVE_IMAGE_PATH):
        print(f"Error: The image file '{LIVE_IMAGE_PATH}' was not found after download.")
        print("Please check the `fetch_data.py` script for issues.")
        exit()

    # --- 2. Run the detection on the newly downloaded image ---
    print("\n--- Step 2: Running deforestation detection ---")
    blended_image, mask_image, deforestation_percentage = run_detection_on_image(
        image_path=LIVE_IMAGE_PATH,
        model_path=MODEL_PATH
    )

    if blended_image is None:
        print("Failed to run detection. Please ensure the model file exists and the image is valid.")
        exit()

    # --- 3. Display the Results and Alert ---
    print("\n--- Step 3: Displaying results ---")
    print(f"Deforestation detected: {deforestation_percentage:.2f}%")

    if deforestation_percentage > 0.1:  # You can adjust this threshold
        print("\n🚨🚨🚨 ALERT! Deforestation Detected! 🚨🚨🚨")
        print(f"Deforestation was detected with a change of {deforestation_percentage:.2f}%.")
    else:
        print("\n✅ No significant deforestation detected.")

    # Save the result images for verification
    if not os.path.exists('results'):
        os.makedirs('results')

    blended_image.save('results/blended_result.png')
    mask_image.save('results/mask_result.png')
    print("\nResult images saved to the 'results/' folder.")
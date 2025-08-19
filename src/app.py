# src/app.py

from flask import Flask, render_template, jsonify, request
import os
import sys
import requests
import ee
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the core detection and fetching logic
from deforestation_engine import run_detection_on_image
from fetch_data import fetch_image_for_date_range, fetch_live_image, authenticate_and_initialize_gee

# Initialize the Flask app and define paths
app = Flask(__name__, static_folder='static', template_folder='templates')

# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIVE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'deforestation_alerts')
LIVE_IMAGE_PATH = os.path.join(LIVE_IMAGE_FOLDER, 'latest_satellite_image.tif')
HISTORICAL_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'deforestation_alerts', 'historical')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'deforestation_3band_model.h5')

# Ensure the data directories exist
os.makedirs(LIVE_IMAGE_FOLDER, exist_ok=True)
os.makedirs(HISTORICAL_IMAGE_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/detect-live')
def detect_deforestation_live():
    """
    API endpoint to detect deforestation for a given lat/lon for the latest image.
    1. Fetches the latest satellite image.
    2. Runs the trained model on the image.
    3. Returns the blended image and the percentage.
    """
    # Get parameters from URL
    lat_str = request.args.get('lat')
    lon_str = request.args.get('lon')

    if not lat_str or not lon_str:
        return jsonify({"error": "Latitude and longitude parameters are required."}), 400

    try:
        lat = float(lat_str)
        lon = float(lon_str)
    except ValueError:
        return jsonify({"error": "Invalid latitude or longitude format. Must be a number."}), 400

    # Step 1: Fetch the new live image
    if not authenticate_and_initialize_gee():
        return jsonify({"error": "Failed to authenticate with Earth Engine."}), 500
    
    success_fetch, message_fetch = fetch_live_image(lat, lon)
    if not success_fetch:
        return jsonify({"error": message_fetch}), 500

    # Step 2: Run the detection on the newly downloaded image
    blended_image, mask_image, percentage = run_detection_on_image(
        image_path=LIVE_IMAGE_PATH,
        model_path=MODEL_PATH
    )

    if blended_image is None:
        return jsonify({"error": "Failed to run detection. Check model and image paths."}), 500

    # Step 3: Save the temporary result images for the frontend
    results_dir = os.path.join(app.root_path, 'static', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    blended_path = os.path.join(results_dir, 'live_blended.png')
    mask_path = os.path.join(results_dir, 'live_mask.png')
    
    blended_image.save(blended_path)
    if mask_image:
        mask_image.save(mask_path)

    # Step 4: Return the paths and percentage to the frontend
    return jsonify({
        "success": True,
        "percentage": f"{percentage:.2f}%",
        "blended_image_url": f"/static/results/live_blended.png?_t={datetime.now().timestamp()}",
        "mask_image_url": f"/static/results/live_mask.png?_t={datetime.now().timestamp()}"
    })


@app.route('/detect-historical')
def detect_historical_deforestation():
    """
    API endpoint to detect historical deforestation for a given lat/lon and date range.
    1. Fetches two satellite images for the date range.
    2. Runs the trained model on both images.
    3. Compares the results to get the change percentage.
    4. Returns the results and the percentage.
    """
    # Get parameters from URL
    lat_str = request.args.get('lat')
    lon_str = request.args.get('lon')
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if not lat_str or not lon_str or not start_date_str or not end_date_str:
        return jsonify({"error": "Latitude, longitude, start_date, and end_date parameters are required."}), 400

    try:
        lat = float(lat_str)
        lon = float(lon_str)
    except ValueError:
        return jsonify({"error": "Invalid latitude or longitude format. Must be a number."}), 400

    # Step 1: Authenticate Earth Engine
    if not authenticate_and_initialize_gee():
        return jsonify({"error": "Failed to authenticate with Earth Engine."}), 500

    # Step 2: Fetch the "before" image
    success_initial, initial_image_path = fetch_image_for_date_range(
        lat, lon, start_date_str, 'initial'
    )
    if not success_initial:
        return jsonify({"error": f"Failed to get initial image: {initial_image_path}"}), 500

    # Step 3: Fetch the "after" image
    success_final, final_image_path = fetch_image_for_date_range(
        lat, lon, end_date_str, 'final'
    )
    if not success_final:
        return jsonify({"error": f"Failed to get final image: {final_image_path}"}), 500

    # Step 4: Run detection on both images
    blended_initial, mask_initial, _ = run_detection_on_image(
        image_path=initial_image_path,
        model_path=MODEL_PATH
    )
    blended_final, mask_final, _ = run_detection_on_image(
        image_path=final_image_path,
        model_path=MODEL_PATH
    )

    if blended_initial is None or blended_final is None or mask_initial is None or mask_final is None:
        return jsonify({"error": "Failed to run detection on historical images."}), 500

    # Step 5: Compare the masks to get the percentage of new deforestation
    # FIX: Convert PIL Images to NumPy arrays before comparison
    initial_mask_array = np.array(mask_initial.convert('L'))
    final_mask_array = np.array(mask_final.convert('L'))

    # Binarize the arrays
    initial_mask_binarized = (initial_mask_array > 0).astype(int)
    final_mask_binarized = (final_mask_array > 0).astype(int)
    
    # Calculate new deforestation (pixels that are 0 in initial and 1 in final)
    # Use np.logical_and for clarity and to get a boolean mask
    new_deforestation_mask = np.logical_and(initial_mask_binarized == 0, final_mask_binarized == 1)
    
    new_deforestation_pixels = np.sum(new_deforestation_mask)
    total_pixels = new_deforestation_mask.size
    percentage = (new_deforestation_pixels / total_pixels) * 100

    # Step 6: Save the temporary result images for the frontend
    results_dir = os.path.join(app.root_path, 'static', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    initial_blended_path = os.path.join(results_dir, 'initial_blended.png')
    final_blended_path = os.path.join(results_dir, 'final_blended.png')
    
    blended_initial.save(initial_blended_path)
    blended_final.save(final_blended_path)

    # Step 7: Return the paths and percentage to the frontend
    return jsonify({
        "success": True,
        "percentage": f"{percentage:.2f}%",
        "initial_blended_image_url": f"/static/results/initial_blended.png?_t={datetime.now().timestamp()}",
        "final_blended_image_url": f"/static/results/final_blended.png?_t={datetime.now().timestamp()}"
    })


if __name__ == '__main__':
    # Initial authentication for Earth Engine
    if authenticate_and_initialize_gee():
        # Start the Flask app
        app.run(debug=True)

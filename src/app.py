# src/app.py

import os
import io
from flask import Flask, request, render_template, send_file, jsonify
from deforestation_engine import run_detection_on_image, run_detection_on_comparison
from fetch_data import fetch_deforestation_data, fetch_sar_data, combine_data
import ee
from datetime import datetime, timedelta

# Import the core detection and fetching logic
from deforestation_engine import run_detection_on_comparison, run_detection_on_image
from fetch_data import fetch_image_for_date_range, fetch_live_image, authenticate_and_initialize_gee

# Initialize the Flask app and define paths
app = Flask(__name__, static_folder='static', template_folder='templates')

# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIVE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'deforestation_alerts')
LIVE_IMAGE_PATH = os.path.join(LIVE_IMAGE_FOLDER, 'latest_satellite_image.tif')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'deforestation_3band_model.h5')


@app.route('/')
def index():
    """Main route for the dashboard."""
    return render_template('index.html')

@app.route('/historical')
def historical():
    """Route for the historical deforestation detection page."""
    return render_template('historical.html')


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
    
    blended_image.save(blended_path)

    # Step 4: Return the paths and percentage to the frontend
    return jsonify({
        "success": True,
        "percentage": f"{percentage:.2f}%",
        "blended_image_url": f"/static/results/live_blended.png?_t={datetime.now().timestamp()}"
    })


@app.route('/detect')
def detect_historical_deforestation():
    """
    API endpoint to detect deforestation for a given lat/lon and date range.
    1. Fetches two satellite images for the date range.
    2. Runs the trained model on both images.
    3. Compares the results and returns the percentage of new deforestation.
    """
    # Get parameters from URL
    lat_str = request.args.get('lat')
    lon_str = request.args.get('lon')
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if not lat_str or not lon_str or not start_date_str or not end_date_str:
        return jsonify({"error": "Latitude, longitude, start_date, and end_date parameters are required."}), 400

    try:
        # Define a small bounding box around the point
        bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
        
        # Fetch and combine data for the initial period
        s2_initial = fetch_deforestation_data(bbox, initial_date_str, initial_date_str)
        s1_initial = fetch_sar_data(bbox, initial_date_str, initial_date_str)
        combined_initial = combine_data(s2_initial, s1_initial)

    # Step 2: Fetch two images for the date range
    # Fix: Add the 'image_type' argument
    success_initial, initial_image_path = fetch_image_for_date_range(
        lat, lon, start_date_str, start_date_str, 'initial'
    )
    if not success_initial:
        return jsonify({"error": initial_image_path}), 500

    # Fix: Add the 'image_type' argument
    success_final, final_image_path = fetch_image_for_date_range(
        lat, lon, end_date_str, end_date_str, 'final'
    )
    if not success_final:
        return jsonify({"error": final_image_path}), 500
    
    # Step 3: Run the comparison detection
    blended_image, mask_image, percentage = run_detection_on_comparison(
        initial_image_path=initial_image_path,
        final_image_path=final_image_path,
        model_path=MODEL_PATH
    )

    if blended_image is None:
        return jsonify({"error": "Failed to run detection. Check model and image paths."}), 500

    # Step 4: Save the temporary result images for the frontend
    results_dir = os.path.join(app.root_path, 'static', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    blended_path = os.path.join(results_dir, 'blended.png')
    mask_path = os.path.join(results_dir, 'mask.png')
    
    blended_image.save(blended_path)
    mask_image.save(mask_path)

    # Step 5: Return the paths and percentage to the frontend
    return jsonify({
        "success": True,
        "percentage": f"{percentage:.2f}%",
        "blended_image_url": f"/static/results/blended.png?_t={datetime.now().timestamp()}",
        "mask_image_url": f"/static/results/mask.png?_t={datetime.now().timestamp()}"
    })


if __name__ == '__main__':
    # You might want to run in debug mode for development
    app.run(debug=True, port=5000)
# src/app.py

from flask import Flask, render_template, jsonify, request
import os
import sys

# Import the core detection logic and data fetching logic
from deforestation_engine import run_detection_on_image
from fetch_data import fetch_live_image, authenticate_and_initialize_gee

# Initialize the Flask app and define paths
app = Flask(__name__, static_folder='static', template_folder='templates')

# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIVE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'deforestation_alerts')
LIVE_IMAGE_PATH = os.path.join(LIVE_IMAGE_FOLDER, 'latest_satellite_image.tif')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'deforestation_3band_model.h5')

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deforestation_api():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required."}), 400

    # Step 1: Authenticate with Earth Engine and fetch the new live image
    if not authenticate_and_initialize_gee():
        return jsonify({"error": "Failed to authenticate with Earth Engine."}), 500
    
    if not fetch_live_image(lat, lon):
        return jsonify({"error": "Failed to download a new satellite image."}), 500

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
    
    blended_path = os.path.join(results_dir, 'blended.png')
    mask_path = os.path.join(results_dir, 'mask.png')
    
    blended_image.save(blended_path)
    mask_image.save(mask_path)

    # Step 4: Return the paths and percentage to the frontend
    return jsonify({
        "success": True,
        "blended_image_url": f"/static/results/blended.png?_={os.path.getmtime(blended_path)}",
        "mask_image_url": f"/static/results/mask.png?_={os.path.getmtime(mask_path)}",
        "deforestation_percentage": round(percentage, 2),
        "alert_message": "Deforestation detected!" if percentage > 0.1 else "No significant deforestation detected."
    })

if __name__ == '__main__':
    # For local development, set debug=True
    app.run(debug=True, port=5000)
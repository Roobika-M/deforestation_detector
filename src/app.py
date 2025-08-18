# src/app.py

from flask import Flask, render_template, jsonify, request
import os
import sys
import requests
import ee
from datetime import datetime, timedelta

# Import the core detection logic
from deforestation_engine import run_detection_on_image

# Initialize the Flask app and define paths
app = Flask(__name__, static_folder='static', template_folder='templates')

# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIVE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'deforestation_alerts')
LIVE_IMAGE_PATH = os.path.join(LIVE_IMAGE_FOLDER, 'latest_satellite_image.tif')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'deforestation_3band_model.h5')

# --- Google Earth Engine Authentication ---
def authenticate_and_initialize_gee():
    """Authenticates and initializes the Google Earth Engine API."""
    try:
        # Use your specific Google Cloud Project ID
        GOOGLE_CLOUD_PROJECT_ID = 'amazing-math-417115' 
        ee.Initialize(project=GOOGLE_CLOUD_PROJECT_ID)
        print("Earth Engine initialized successfully.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please ensure you have authenticated with 'earthengine authenticate' and your Project ID is correct.")
        return False
    return True

# New function to download with retries
def download_file(url, local_path, retries=5):
    """Downloads a file from a URL with retry logic."""
    for i in range(retries):
        try:
            print(f"Attempting to download {os.path.basename(local_path)} (Attempt {i+1}/{retries})...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Download successful: {local_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            if i < retries - 1:
                print("Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                print("❌ Max retries exceeded. Download failed.")
                return False
    return False

def fetch_live_image(lat, lon):
    """Fetches a live Sentinel-2 image for a given coordinate."""
    print("Starting to fetch input image for live detection...")
    
    # Define a small Area of Interest (AOI) around the given coordinates
    aoi_coords = [
        [lon - 0.02, lat - 0.02],
        [lon + 0.02, lat - 0.02],
        [lon + 0.02, lat + 0.02],
        [lon - 0.02, lat + 0.02],
        [lon - 0.02, lat - 0.02]
    ]
    aoi = ee.Geometry.Polygon(aoi_coords)

    # Fetch the latest, cloud-free image
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    sentinel_image = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .filterBounds(aoi)
        .sort('system:time_start', False)
        .first()
    )

    if not sentinel_image:
        print("No suitable Sentinel-2 images found.")
        return False

    sentinel_image = sentinel_image.select(['B4', 'B3', 'B2'])
    
    try:
        sentinel_url = sentinel_image.getDownloadUrl({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': aoi.getInfo()['coordinates'],
            'format': 'GEO_TIFF'
        })
    except ee.EEException as e:
        print(f"Error getting download URL: {e}")
        return False

    return download_file(sentinel_url, LIVE_IMAGE_PATH)


@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/detect', methods=['GET'])
def detect_deforestation():
    """
    API endpoint to trigger the deforestation detection.
    This now accepts latitude and longitude from the frontend.
    """
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude are required."}), 400

    # Step 1: Fetch the new live image
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
        "percentage": f"{percentage:.2f}%",
        "blended_image_url": f"/static/results/blended.png?ts={datetime.now().timestamp()}",
        "mask_image_url": f"/static/results/mask.png?ts={datetime.now().timestamp()}"
    })

if __name__ == '__main__':
    app.run(debug=True)

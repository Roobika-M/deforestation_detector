# src/fetch_data.py

import ee
import requests
import os
import sys
from datetime import datetime, timedelta
import ee.mapclient

# Use the same paths as defined in app.py for consistency
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIVE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'deforestation_alerts')
INPUT_IMAGE_PATH = os.path.join(LIVE_IMAGE_FOLDER, 'latest_satellite_image.tif')

# --- Google Earth Engine Authentication ---
def authenticate_and_initialize_gee():
    """Authenticates and initializes the Google Earth Engine API."""
    try:
        GOOGLE_CLOUD_PROJECT_ID = 'amazing-math-417115'
        ee.Initialize(project=GOOGLE_CLOUD_PROJECT_ID)
        print("Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please ensure you have authenticated with 'earthengine authenticate' and your Project ID is correct.")
        return False

# New function to download with retries
def download_file(url, local_path, retries=5):
    """Downloads a file from a URL with retry logic."""
    for i in range(retries):
        try:
            print(f"Attempting to download {os.path.basename(local_path)} (Attempt {i+1}/{retries})...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download successful.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            if i < retries - 1:
                print("Retrying...")
            else:
                print("Max retries reached. Download failed.")
                return False

def fetch_live_image(lat, lon):
    """
    Fetches the latest Sentinel-2 image for a given latitude and longitude.
    
    Args:
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center point.
        
    Returns:
        bool: True if the image was successfully downloaded, False otherwise.
    """
    print("Starting to fetch input image for live detection...")

    # Define a small square Area of Interest (AOI) around the lat/lon
    # 0.01 degrees is roughly 1.1 km
    aoi = ee.Geometry.Rectangle([lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01])

    # --- Fetch the Latest Sentinel-2 Image (Input Data) ---
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
        print("No suitable Sentinel-2 images found in the date range. Please try a different AOI or date window.")
        return False

    # Select the B4 (Red), B3 (Green), and B2 (Blue) bands
    sentinel_image = sentinel_image.select(['B4', 'B3', 'B2'])
    
    try:
        sentinel_url = sentinel_image.getDownloadUrl({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': aoi.getInfo()['coordinates'],
            'format': 'GEO_TIFF'
        })
    except ee.EEException as e:
        print(f"Error getting download URL from Earth Engine: {e}")
        return False
        
    # Create the directory if it doesn't exist
    os.makedirs(LIVE_IMAGE_FOLDER, exist_ok=True)
    
    # Download the image
    return download_file(sentinel_url, INPUT_IMAGE_PATH)
import ee
import requests
import os
import sys
from datetime import datetime, timedelta

# --- Configuration ---
# Replace with your specific Google Cloud Project ID
GOOGLE_CLOUD_PROJECT_ID = 'amazing-math-417115'

# Using a new AOI for live detection (e.g., in Bolivia)
AOI_COORDINATES = [
    [-63.02, -14.99],
    [-62.97, -14.99],
    [-62.97, -15.02],
    [-63.02, -15.02],
    [-63.02, -14.99]
]

# Define the folder and filename for the live image
DATA_FOLDER = '../deforestation_alerts'
INPUT_IMAGE_PATH = os.path.join(DATA_FOLDER, 'latest_satellite_image.tif')
# Labels are not needed for live detection, so this path is not used
LABELS_PATH = None 

# New function to download with retries
def download_file(url, local_path, retries=5):
    """Downloads a file from a URL with retry logic."""
    for i in range(retries):
        try:
            print(f"Attempting to download {os.path.basename(local_path)} (Attempt {i+1}/{retries})...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

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

def authenticate_and_initialize_gee():
    """Authenticates and initializes the Google Earth Engine API with a specific project."""
    try:
        ee.Initialize(project=GOOGLE_CLOUD_PROJECT_ID)
        print("Earth Engine initialized successfully.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please run 'earthengine authenticate' in your terminal and check your Project ID.")
        sys.exit(1)

def fetch_and_save_data():
    """Fetches the latest Sentinel-2 image for live detection."""
    print("Starting to fetch input image for live detection...")

    aoi = ee.Geometry.Polygon(AOI_COORDINATES)

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

    sentinel_image = sentinel_image.select(['B4', 'B3', 'B2'])
    
    # Get download URL for the sentinel image only
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

    # --- Download the file ---
    os.makedirs(DATA_FOLDER, exist_ok=True)
    input_success = download_file(sentinel_url, INPUT_IMAGE_PATH)

    return input_success

if __name__ == '__main__':
    authenticate_and_initialize_gee()
    fetch_and_save_data()
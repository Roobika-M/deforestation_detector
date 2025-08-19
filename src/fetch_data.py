import ee
import os
import sys
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the folder and filename for the live image
DATA_FOLDER = '../deforestation_alerts'
LIVE_IMAGE_PATH = os.path.join(DATA_FOLDER, 'latest_satellite_image.tif')
HISTORICAL_IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'historical')

def authenticate_and_initialize_gee():
    """Authenticates and initializes the Google Earth Engine API."""
    try:
        # Use your specific Google Cloud Project ID
        GOOGLE_CLOUD_PROJECT_ID = 'amazing-math-417115'
        ee.Initialize(project=GOOGLE_CLOUD_PROJECT_ID)
        logging.info("Earth Engine initialized successfully.")
        return True
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        return False

def download_file(url, local_path, retries=5):
    """Downloads a file from a URL with retry logic."""
    for i in range(retries):
        try:
            logging.info(f"Attempting to download {os.path.basename(local_path)} (Attempt {i+1}/{retries})...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Ensure the directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.info(f"✅ Download successful: {local_path}")
            return True, "Download successful."
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Download failed: {e}")
            if i < retries - 1:
                logging.info("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logging.error(f"❌ Max retries exceeded. Download failed.")
                return False, f"Download failed after {retries} retries."
    return False, "Download process did not complete."

def fetch_live_image(lat, lon):
    """
    Fetches the latest Sentinel-2 image for a given latitude and longitude.
    """
    if not authenticate_and_initialize_gee():
        return False, "Failed to authenticate with Earth Engine."

    # Define the area of interest (AOI) as a point
    aoi = ee.Geometry.Point(lon, lat).buffer(1000)

    # Load Sentinel-2 surface reflectance images, filter by bounds and date,
    # and apply the cloud mask.
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .map(mask_clouds_and_shadows) \
        .median() # Take the median to create a single composite image

    return collection
    
def fetch_sar_data(bbox, start_date, end_date):
    """
    Fetches and processes Sentinel-1 SAR data from Google Earth Engine.
    Args:
        bbox: A list of coordinates [min_lon, min_lat, max_lon, max_lat]
              defining the bounding box.
        start_date: Start date for the image collection.
        end_date: End date for the image collection.
    Returns:
        An ee.ImageCollection of processed Sentinel-1 imagery.
    """
    if not authenticate_and_initialize_gee():
        return False, "Failed to authenticate with Earth Engine."

    try:
        # Define the date as the center of a search window
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        search_start_date = target_date - timedelta(days=3)
        search_end_date = target_date + timedelta(days=3)
        logging.info(f"Attempting to find a {image_type} image between {search_start_date.strftime('%Y-%m-%d')} and {search_end_date.strftime('%Y-%m-%d')}")
    except ValueError as e:
        return False, f"Invalid date format: {e}. Please use YYYY-MM-DD."

    # Define the area of interest (AOI) as a point
    aoi = ee.Geometry.Point(lon, lat).buffer(1000)

    # Fetch the image for the specified date range
    sentinel_image = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(search_start_date, search_end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
        .filterBounds(aoi)
        .sort('system:time_start', False)
        .first()
    )

    if not sentinel_image.getInfo():
        return False, f"No suitable Sentinel-2 images found for the {image_type} period ({date_str}). Please try different dates."

    # Select the RGB bands
    sentinel_image = sentinel_image.select(['B4', 'B3', 'B2'])

    try:
        sentinel_url = sentinel_image.getDownloadUrl({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': aoi.getInfo()['coordinates'],
            'format': 'GEO_TIFF'
        })
    except ee.EEException as e:
        return False, f"Error generating download URL: {e}"
        
    # Define the local path for the historical image
    local_path = os.path.join(HISTORICAL_IMAGE_FOLDER, f'satellite_image_{image_type}.tif')

    # Download the image
    success, message = download_file(sentinel_url, local_path)
    
    if not success:
        return False, message
    
    return True, local_path

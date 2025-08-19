import ee
import os
import geemap

# Initialize the Earth Engine connection
# Note: You need to have authenticated and initialized EE previously
# e.g., ee.Authenticate() and ee.Initialize()
# We assume this is done in a main script or a setup file.

def mask_clouds_and_shadows(image):
    """
    Masks clouds and shadows in a Sentinel-2 SR image.
    This function is adapted from a Google Earth Engine community example.
    """
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    # Return the masked image, with the 'cloud probability' band as well
    # for potential future use.
    return image.updateMask(mask).divide(10000)

def fetch_deforestation_data(bbox, start_date, end_date):
    """
    Fetches deforestation data (Sentinel-2 SR images) from Google Earth Engine.
    Args:
        bbox: A list of coordinates [min_lon, min_lat, max_lon, max_lat]
              defining the bounding box.
        start_date: Start date for the image collection (e.g., '2023-01-01').
        end_date: End date for the image collection (e.g., '2023-01-31').
    Returns:
        An ee.ImageCollection with cloud-masked Sentinel-2 imagery.
    """
    # Create a geometry from the bounding box
    geometry = ee.Geometry.Rectangle(bbox)

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
    # Create a geometry from the bounding box
    geometry = ee.Geometry.Rectangle(bbox)

    # Load Sentinel-1 GRD imagery, filter by bounds, date, and polarization
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

    # Apply a speckle filter to reduce noise
    def apply_speckle_filter(image):
        return image.focal_median()

    # Apply the filter and take the median composite
    sar_image = collection.map(apply_speckle_filter).median()
    
    # We only need the 'VV' band for this analysis
    return sar_image.select('VV')

def combine_data(s2_image, s1_image):
    """
    Combines Sentinel-2 and Sentinel-1 images into a single image.
    Args:
        s2_image: The processed Sentinel-2 image.
        s1_image: The processed Sentinel-1 image.
    Returns:
        A combined ee.Image with bands from both satellites.
    """
    # Resize SAR image to match Sentinel-2 resolution
    s1_resized = s1_image.reproject(crs=s2_image.select('B4').projection())
    
    # Combine the bands from both images
    return s2_image.addBands(s1_resized)

# After your import statements and before any EE calls
if __name__ == '__main__':
    # Add your Project ID here
    project_id = 'amazing-math-417115'
    
    # Authenticate and initialize Earth Engine with project ID
    try:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("Earth Engine initialized successfully.")
    except ee.ee_exception.EEException as e:
        print(f"Error initializing Earth Engine: {e}")
        print("Please ensure you have authenticated and are using a valid project ID.")
        exit()

    # Example usage:
    example_bbox = [-68.8, -10.5, -68.4, -10.2]
    start_date = '2023-01-01'
    end_date = '2023-03-31'
    # ... the rest of your code
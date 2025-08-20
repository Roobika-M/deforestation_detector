import ee
import json
import requests
from io import BytesIO
from PIL import Image

# --- Initialize Earth Engine ---
def ee_init():
    try:
        ee.Initialize(project='amazing-math-417115')  
    except Exception:
        print("Re-authenticating...")
        ee.Authenticate()
        ee.Initialize(project='amazing-math-417115')


# --- Convert bbox to AOI ---
def bbox_to_aoi(min_lon, min_lat, max_lon, max_lat):
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


# --- Helper: Cloud-masked Sentinel-2 collection ---
def _s2_sr_collection(aoi, start_date, end_date, cloud_pct=40):
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
    )

    def mask_clouds(img):
        qa = img.select('QA60')
        cloud_bit_mask = (1 << 10)
        cirrus_bit_mask = (1 << 11)
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
               qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return img.updateMask(mask).divide(10000)

    return col.map(mask_clouds)


# --- Fetch median RGB as PIL image ---
def get_s2_rgb(aoi, start_date, end_date, cloud_pct=40):
    col = _s2_sr_collection(aoi, start_date, end_date, cloud_pct)
    size = col.size().getInfo()
    if size == 0:
        raise ValueError(f"No Sentinel-2 images found for {start_date} - {end_date}")

    img = col.median().select(['B4', 'B3', 'B2'])
    url = img.getThumbURL({
        "region": json.loads(aoi.toGeoJSONString()),
        "dimensions": 512,
        "min": 0,
        "max": 0.3
    })
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

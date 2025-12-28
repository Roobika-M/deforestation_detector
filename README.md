# Forest Monitor – AI Deforestation Detector

Forest Monitor is a two-part application that fetches cloud-masked Sentinel-2 imagery from Google Earth Engine, applies a U-Net segmentation model to highlight forest cover, and compares time windows to flag potential deforestation. A lightweight frontend offers live checks, historical comparisons, and background monitoring against a Flask API.

## Features
- Live check: compare the last 30 days against an optional baseline period.
- Historical comparison: supply two custom date ranges to quantify canopy change.
- Continuous monitoring: background loop that re-checks an AOI at a chosen interval and raises alerts when loss exceeds a threshold.
- Visual overlays: AI masks blended onto RGB imagery with quick stats (total pixels and loss percentage).

## Project Structure
- backend/: Flask API, Earth Engine fetch, inference utilities, U-Net checkpoint (unet_forest.pth).
- frontend/: Single-page UI (HTML/CSS/JS) that calls the API and renders results.
- model/ and training/: assets and notebook used during model training (not required to run the app).

## Requirements
- Python 3.10+ recommended.
- Google Earth Engine access; authentication will prompt on first run.
- PyTorch with CUDA if you want GPU acceleration; CPU works but is slower.
- U-Net weights file `backend/unet_forest.pth` (place alongside the backend code).

Python dependencies (key ones):
- flask, flask-cors
- pillow, numpy
- earthengine-api, requests
- torch, torchvision
- segmentation-models-pytorch
- scipy

> Note: `backend/requirements.txt` lists the core packages but you may need to add `torch`, `torchvision`, `segmentation-models-pytorch`, and `scipy` according to your environment (CUDA vs CPU). Install PyTorch from https://pytorch.org/get-started/locally/ for the right wheels.

## Setup
1) Create and activate a virtual environment.
2) Install dependencies (adjust the torch command per your platform):
   ```bash
   pip install -r backend/requirements.txt
   pip install torch torchvision segmentation-models-pytorch scipy
   ```
3) Place `unet_forest.pth` inside `backend/`.
4) Authenticate Earth Engine on first run; the app calls `ee.Authenticate()` if needed. Ensure the project ID in `backend/gee_fetch.py` (`amazing-math-417115`) matches your GEE project or update it.

## Running the app
From the backend directory:
```bash
python app.py
```
This starts Flask on http://127.0.0.1:5000 and serves the frontend from `/frontend`.

## API Overview
- `GET /api/health` – basic health info.
- `POST /api/gee/live` – live comparison.
  - Body: `{ "bbox": [minLon, minLat, maxLon, maxLat], "baseline": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD" } }`
  - If baseline is omitted, it defaults to a 30-day window from one year ago; the "after" window is the last 30 days.
- `POST /api/gee/change` – historical comparison.
  - Body: `{ "bbox": [...], "before": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}, "after": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} }`
- `POST /api/monitor/start` – begin background monitoring.
  - Body: `{ "bbox": [...], "interval": 3600, "threshold": 20.0 }` (interval seconds, threshold percent loss)
- `POST /api/monitor/stop` – stop monitoring.

The response for comparison endpoints includes `images.before`, `images.after` (base64 PNG overlays), `stats.loss_pct`, and an `alert` flag with an `alert_reason` string.

## Bounding box format
All endpoints expect `[minLon, minLat, maxLon, maxLat]`. Example: `[-60, -10, -59, -9]`.

## Frontend usage
Open the served page (root path). Use the sidebar tabs to:
- Live Check: run quick analysis for the last 30 days, optionally providing a baseline range.
- Historical: provide two date ranges to compare forest cover.
- Monitoring: start/stop the periodic checker with a custom interval.

## Notes
- Cloud masking is applied to Sentinel-2 SR imagery; missing imagery for a period returns an HTTP 404 with a message.
- Alerting is based on pixel-level loss percentage; detections above 60% are treated as unreliable (possible clouds/seasonality) and suppressed.
- Resize and overlay helpers are in `backend/utils.py`; model inference is defined in `backend/inference.py` using a ResNet-34 U-Net from segmentation-models-pytorch.

## Troubleshooting
- Authentication errors: run a small Python snippet `import ee; ee.Authenticate()` and follow the browser flow, or delete cached tokens and retry.
- Model missing: ensure `backend/unet_forest.pth` is present; the app will raise `FileNotFoundError` otherwise.
- Slow inference: install a CUDA-enabled PyTorch build and set `device="cuda"` when initializing `UNetInference` if modifying the code.

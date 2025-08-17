# app.py

from flask import Flask, render_template, jsonify, send_file
import os
from datetime import datetime, timedelta

# Import the new detection function
from deforestation_engine import run_detection_on_image

app = Flask(__name__)

# --- Configuration ---
LIVE_IMAGE_FOLDER = '../deforestation_alerts'
LIVE_IMAGE_PATH = os.path.join(LIVE_IMAGE_FOLDER, 'latest_satellite_image.tif')

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/detect', methods=['GET'])
def detect_deforestation():
    """
    API endpoint to trigger the deforestation detection.
    This simulates fetching a new image and running the model.
    """
    # In a real app, you would run the fetch_data.py logic here
    # For this example, we assume the file is already downloaded.
    if not os.path.exists(LIVE_IMAGE_PATH):
        return jsonify({"error": "No satellite image found. Please run fetch_data.py first."}), 404

    # Run the detection using the function from your engine file
    blended_image, mask_image, percentage = run_detection_on_image(LIVE_IMAGE_PATH)

    # Save the temporary result images to a folder for the frontend to access
    results_dir = os.path.join(app.root_path, 'static', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    blended_path = os.path.join(results_dir, 'blended.png')
    mask_path = os.path.join(results_dir, 'mask.png')
    
    blended_image.save(blended_path)
    mask_image.save(mask_path)

    # Return the paths and percentage to the frontend
    return jsonify({
        "success": True,
        "percentage": f"{percentage:.2f}%",
        "blended_image_url": f"/static/results/blended.png?ts={datetime.now().timestamp()}",
        "mask_image_url": f"/static/results/mask.png?ts={datetime.now().timestamp()}"
    })

if __name__ == '__main__':
    # Add your `fetch_data.py` logic to this file if you want a fully automated process
    app.run(debug=True)
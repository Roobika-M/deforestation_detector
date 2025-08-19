# src/app.py

import os
import io
from flask import Flask, request, render_template, send_file, jsonify
from deforestation_engine import run_detection_on_image, run_detection_on_comparison
from fetch_data import fetch_deforestation_data, fetch_sar_data, combine_data
import ee
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path to your trained model file.
# Make sure to update this path after training your new 4-band model.
MODEL_PATH = 'deforestation_4band_model.h5'

# Initialize Earth Engine with your Project ID
try:
    # Replace 'your-project-id' with your actual Google Cloud Project ID
    ee.Initialize(project='your-project-id')
    print("Earth Engine initialized successfully.")
except ee.EEException as e:
    print(f"Error initializing Earth Engine: {e}")
    print("Please ensure you have authenticated and are using a valid project ID.")
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/historical')
def historical():
    return render_template('historical.html')
    
@app.route('/api/detect-live', methods=['POST'])
def detect_live():
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    date_str = data.get('date')
    
    if not all([lat, lon, date_str]):
        return jsonify({"error": "Missing latitude, longitude, or date"}), 400

    try:
        # Define a small bounding box around the point
        bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
        
        # We need a start and end date for GEE. Use a 1-month range.
        start_date = date_str
        end_date = ee.Date(date_str).advance(1, 'month').format('YYYY-MM-dd').getInfo()

        # Fetch and combine the data
        s2_image = fetch_deforestation_data(bbox, start_date, end_date)
        s1_image = fetch_sar_data(bbox, start_date, end_date)
        combined_image = combine_data(s2_image, s1_image)
        
        # Save the combined image temporarily
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'live_image.tif')
        
        # The ee.Image needs to be exported to a local file before model inference.
        # This is a simplified approach. A real-world app would use more robust export.
        ee.batch.Export.image.toDrive(image=combined_image,
                                      description='live_image',
                                      folder='deforestation_app',
                                      fileNamePrefix='live_image',
                                      scale=10, # Sentinel-2 resolution
                                      region=ee.Geometry.Rectangle(bbox)).start()
                                      
        # Note: Export to Google Drive is an asynchronous process.
        # You would need a more complex system to wait for this to complete.
        # For this project, you can assume the file is there or handle the delay.
        
        # For simplicity in this example, we assume you have a local TIFF file already.
        # Replace the above with a manual export and place the TIFF in the uploads folder.
        # For example: temp_image_path = 'uploads/manually_exported_image.tif'
        
        # Since we can't wait for the export, we'll return a message to the user.
        return jsonify({
            "message": "Data export to Google Drive started. Please wait for the file to be ready.",
            "status": "pending"
        })

        # --- The following code is for after you have the local TIFF file ---
        # blended_image, mask_image, deforestation_percent = run_detection_on_image(temp_image_path, MODEL_PATH)

        # if blended_image is None:
        #     return jsonify({"error": "Failed to run detection on the image."}), 500

        # # Convert PIL images to bytes to send back to the client
        # blended_img_io = io.BytesIO()
        # blended_image.save(blended_img_io, format='PNG')
        # blended_img_io.seek(0)

        # mask_img_io = io.BytesIO()
        # mask_image.save(mask_img_io, format='PNG')
        # mask_img_io.seek(0)
        
        # return jsonify({
        #     "blended_image": blended_img_io.getvalue().decode('latin-1'),
        #     "mask_image": mask_img_io.getvalue().decode('latin-1'),
        #     "deforestation_percent": deforestation_percent
        # })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect-historical', methods=['POST'])
def detect_historical():
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    initial_date_str = data.get('initial_date')
    final_date_str = data.get('final_date')
    
    if not all([lat, lon, initial_date_str, final_date_str]):
        return jsonify({"error": "Missing latitude, longitude, or dates"}), 400

    try:
        # Define a small bounding box around the point
        bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
        
        # Fetch and combine data for the initial period
        s2_initial = fetch_deforestation_data(bbox, initial_date_str, initial_date_str)
        s1_initial = fetch_sar_data(bbox, initial_date_str, initial_date_str)
        combined_initial = combine_data(s2_initial, s1_initial)

        # Fetch and combine data for the final period
        s2_final = fetch_deforestation_data(bbox, final_date_str, final_date_str)
        s1_final = fetch_sar_data(bbox, final_date_str, final_date_str)
        combined_final = combine_data(s2_final, s1_final)
        
        # Save the combined images temporarily
        initial_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'initial_image.tif')
        final_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final_image.tif')
        
        # Again, this is an asynchronous process. For a real app, you would need
        # to wait for these files to be ready before calling run_detection_on_comparison.
        # A simple way for a local project is to manually export the files.
        ee.batch.Export.image.toDrive(image=combined_initial,
                                      description='initial_image',
                                      folder='deforestation_app',
                                      fileNamePrefix='initial_image',
                                      scale=10,
                                      region=ee.Geometry.Rectangle(bbox)).start()
        
        ee.batch.Export.image.toDrive(image=combined_final,
                                      description='final_image',
                                      folder='deforestation_app',
                                      fileNamePrefix='final_image',
                                      scale=10,
                                      region=ee.Geometry.Rectangle(bbox)).start()
                                      
        return jsonify({
            "message": "Data export to Google Drive started for historical analysis. Please wait for the files to be ready.",
            "status": "pending"
        })
        
        # --- The following code is for after you have the local TIFF files ---
        # blended_image, mask_image, deforestation_percent = run_detection_on_comparison(
        #     initial_image_path, final_image_path, MODEL_PATH
        # )

        # if blended_image is None:
        #     return jsonify({"error": "Failed to run detection."}), 500

        # # Convert PIL images to bytes to send back to the client
        # blended_img_io = io.BytesIO()
        # blended_image.save(blended_img_io, format='PNG')
        # blended_img_io.seek(0)
        
        # mask_img_io = io.BytesIO()
        # mask_image.save(mask_img_io, format='PNG')
        # mask_img_io.seek(0)
        
        # return jsonify({
        #     "blended_image": blended_img_io.getvalue().decode('latin-1'),
        #     "mask_image": mask_img_io.getvalue().decode('latin-1'),
        #     "deforestation_percent": deforestation_percent
        # })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # You might want to run in debug mode for development
    app.run(debug=True, port=5000)
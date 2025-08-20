import os
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from scipy.ndimage import binary_closing
import threading
import time
import logging

from inference import UNetInference
from utils import pil_to_base64, overlay_mask
from gee_fetch import ee_init, get_s2_rgb, bbox_to_aoi

# ---- Paths ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "unet_forest.pth")

# ---- Flask app ----
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/")
CORS(app)

# ---- Initialize Earth Engine ----
ee_init()

# ---- Load AI model ----
print("Loading U-Net model...")
inference_model = UNetInference(MODEL_PATH)
print("Model loaded successfully.")

# ---- Monitoring Globals ----
monitoring_thread = None
monitoring_active = False


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return {
        "status": "ok",
        "earth_engine": True,
        "model_loaded": True,
        "monitoring": monitoring_active,
    }


def _compare_and_respond(rgb_before, rgb_after, mode: str):
    """Run AI segmentation, compare masks, and return JSON."""
    mask_b, _ = inference_model.predict_mask(rgb_before, threshold=0.7)
    mask_a, _ = inference_model.predict_mask(rgb_after, threshold=0.7)

    mask_b = binary_closing(mask_b, iterations=2)
    mask_a = binary_closing(mask_a, iterations=2)

    total_pixels = int(mask_b.size)
    lost_pixels = int(((mask_b == 1) & (mask_a == 0)).sum())
    loss_pct = (lost_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    overlay_b = overlay_mask(rgb_before, mask_b)
    overlay_a = overlay_mask(rgb_after, mask_a)

    if loss_pct > 60:
        alert = False
        alert_reason = f"⚠️ Unreliable detection (possible cloud/season effect, {loss_pct:.2f}%)"
    else:
        alert = bool(loss_pct > 20)
        alert_reason = f"Deforestation detected: {float(loss_pct):.2f}% loss" if alert else "No major deforestation"

    return {
        "ok": True,
        "mode": mode,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stats": {"pixels": total_pixels, "loss_pct": round(float(loss_pct), 2)},
        "alert": alert,
        "alert_reason": alert_reason,
        "images": {
            "before": pil_to_base64(overlay_b),
            "after": pil_to_base64(overlay_a),
        }
    }


# ---- Historical comparison route ----
@app.route("/api/gee/change", methods=["POST"])
def gee_change():
    data = request.get_json(force=True)
    bbox = data.get("bbox")
    before = data.get("before")
    after = data.get("after")
    if not bbox or not before or not after:
        return jsonify({"ok": False, "error": "Missing bbox/before/after"}), 400

    aoi = bbox_to_aoi(*bbox)

    try:
        rgb_before = get_s2_rgb(aoi, before["start"], before["end"])
    except ValueError as e:
        return jsonify({"ok": False, "error": f"Before period issue: {str(e)}"}), 404

    try:
        rgb_after = get_s2_rgb(aoi, after["start"], after["end"])
    except ValueError as e:
        return jsonify({"ok": False, "error": f"After period issue: {str(e)}"}), 404

    return jsonify(_compare_and_respond(rgb_before, rgb_after, mode="historical"))


# ---- Live comparison route ----
@app.route("/api/gee/live", methods=["POST"])
def gee_live():
    data = request.get_json(force=True)
    bbox = data.get("bbox")
    baseline = data.get("baseline")
    if not bbox:
        return jsonify({"ok": False, "error": "Missing bbox"}), 400

    aoi = bbox_to_aoi(*bbox)
    today = datetime.utcnow().date()

    after = {"start": (today - timedelta(days=30)).isoformat(), "end": today.isoformat()}

    if not baseline:
        baseline = {
            "start": (today - timedelta(days=365)).isoformat(),
            "end":   (today - timedelta(days=335)).isoformat()
        }

    try:
        rgb_before = get_s2_rgb(aoi, baseline["start"], baseline["end"])
    except ValueError as e:
        return jsonify({"ok": False, "error": f"Baseline period issue: {str(e)}"}), 404

    try:
        rgb_after = get_s2_rgb(aoi, after["start"], after["end"])
    except ValueError as e:
        return jsonify({"ok": False, "error": f"Live period issue: {str(e)}"}), 404

    return jsonify(_compare_and_respond(rgb_before, rgb_after, mode="live"))


# ---- Monitoring Background Loop ----
def monitoring_loop(bbox, interval=3600, threshold=20.0):
    global monitoring_active
    logging.info("🌍 Monitoring loop started...")

    while monitoring_active:
        try:
            aoi = bbox_to_aoi(*bbox)
            today = datetime.utcnow().date()
            after = {"start": (today - timedelta(days=30)).isoformat(), "end": today.isoformat()}
            baseline = {
                "start": (today - timedelta(days=365)).isoformat(),
                "end":   (today - timedelta(days=335)).isoformat()
            }

            rgb_before = get_s2_rgb(aoi, baseline["start"], baseline["end"])
            rgb_after = get_s2_rgb(aoi, after["start"], after["end"])

            result = _compare_and_respond(rgb_before, rgb_after, mode="monitoring")

            if result["alert"] and result["stats"]["loss_pct"] > threshold:
                logging.warning(f"🚨 ALERT: {result['alert_reason']} at {bbox}")

        except Exception as e:
            logging.error(f"Error in monitoring loop: {e}")

        time.sleep(interval)


@app.route("/api/monitor/start", methods=["POST"])
def start_monitoring():
    global monitoring_thread, monitoring_active

    data = request.get_json(force=True)
    bbox = data.get("bbox")
    interval = int(data.get("interval", 3600))
    threshold = float(data.get("threshold", 20.0))

    if not bbox:
        return jsonify({"ok": False, "error": "Missing bbox"}), 400

    if monitoring_active:
        return jsonify({"ok": True, "message": "Monitoring already active"})

    monitoring_active = True
    monitoring_thread = threading.Thread(
        target=monitoring_loop, args=(bbox, interval, threshold), daemon=True
    )
    monitoring_thread.start()

    return jsonify({"ok": True, "message": "Monitoring started"})


@app.route("/api/monitor/stop", methods=["POST"])
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    return jsonify({"ok": True, "message": "Monitoring stopped"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

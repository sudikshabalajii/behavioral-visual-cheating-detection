"""
Backend API (Flask)
===================
RESTful endpoints for the multimodal exam cheating detection system.
Serves both prediction and the frontend exam interface.

Endpoints:
  GET  /           → Serve frontend
  GET  /health     → Health check
  POST /predict    → Single prediction (features or base64 frame)
  POST /predict/batch → Batch prediction
  GET  /flags      → Get all flagged suspicious events
  POST /flags/clear → Clear flagged events
"""

import os
import sys
import json
import base64
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scoring.risk_scorer import RiskScorer, compute_risk_score, classify_risk, get_risk_color
from models.feature_fusion import prepare_realtime_input
from webcam.visual_extractor import VisualExtractor
from webcam.behavior_analyzer import BehaviorAnalyzer
from webcam.feature_utils import GazeTracker

# ─── App Setup ────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_ROOT, "frontend"),
)
CORS(app)

# ─── Global State ─────────────────────────────────────────────────────
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
model = None
scaler = None
encoders = None
feature_names = None
risk_scorer = None
visual_extractor = None
behavior_analyzer = None
gaze_tracker = None


def load_models():
    """Load all saved model artifacts on startup."""
    global model, scaler, encoders, feature_names, risk_scorer
    global visual_extractor, behavior_analyzer, gaze_tracker

    print("[API] Loading model artifacts...")

    try:
        scaler = joblib.load(os.path.join(SAVED_MODELS_DIR, "scaler.pkl"))
        encoders = joblib.load(os.path.join(SAVED_MODELS_DIR, "encoders.pkl"))
        feature_names = joblib.load(os.path.join(SAVED_MODELS_DIR, "feature_names.pkl"))
        print(f"[API] Loaded preprocessing artifacts ({len(feature_names)} features)")
    except FileNotFoundError as e:
        print(f"[API] WARNING: Preprocessing artifacts not found: {e}")
        return

    # Try loading Random Forest (lighter, faster for API)
    try:
        model = joblib.load(os.path.join(SAVED_MODELS_DIR, "random_forest.pkl"))
        print("[API] Loaded Random Forest model")
    except FileNotFoundError:
        # Fallback to LSTM
        try:
            from tensorflow.keras.models import load_model
            model = load_model(os.path.join(SAVED_MODELS_DIR, "cnn_lstm_model.keras"))
            print("[API] Loaded CNN+LSTM model")
        except Exception as e:
            print(f"[API] WARNING: No model found: {e}")
            return

    risk_scorer = RiskScorer(model, scaler, encoders, feature_names)

    # Initialize visual extractor for webcam frame processing
    try:
        visual_extractor = VisualExtractor()
        print("[API] Visual extractor initialized")
    except Exception as e:
        print(f"[API] WARNING: Could not initialize visual extractor: {e}")

    # Initialize behavior analyzer
    behavior_analyzer = BehaviorAnalyzer()
    print("[API] Behavior analyzer initialized")

    # Initialize gaze tracker for temporal analysis
    gaze_tracker = GazeTracker(window_size=30)
    print("[API] Gaze tracker initialized")


# ─── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def serve_frontend():
    """Serve the exam interface."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static frontend files."""
    return send_from_directory(app.static_folder, path)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(feature_names) if feature_names else 0,
        "behavior_analyzer": behavior_analyzer is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single prediction endpoint.

    Accepts JSON with either:
      - "features": list of numeric feature values (pre-processed)
      - "frame": base64-encoded webcam frame (JPEG/PNG)
      - "raw_features": dict of named features (raw, will be processed)

    Returns:
        JSON with prediction, risk_score, risk_level, warnings, eye_metrics
    """
    if risk_scorer is None:
        return jsonify({"error": "Model not loaded. Run training first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        # Option 1: Pre-processed feature array
        if "features" in data:
            features = np.array(data["features"], dtype=np.float32).reshape(1, -1)
            result = risk_scorer.predict_risk(features)
            result["warnings"] = []
            result["eye_metrics"] = {}
            return jsonify(result)

        # Option 2: Base64-encoded webcam frame
        elif "frame" in data:
            if visual_extractor is None:
                return jsonify({"error": "Visual extractor not available"}), 503

            # Decode frame
            frame_data = base64.b64decode(data["frame"])
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return jsonify({"error": "Could not decode frame"}), 400

            # Extract visual features (now includes EAR, iris ratio, gaze velocity)
            frame_features = visual_extractor.extract_features(frame)

            # Update gaze tracker with temporal data
            if gaze_tracker:
                gaze_tracker.update(frame_features)

            # ── Behavior Analysis ─────────────────────────────────
            warnings = []
            if behavior_analyzer:
                warnings = behavior_analyzer.analyze_frame(frame_features)

            # Convert to model input (filter to only trained features)
            features = prepare_realtime_input(
                frame_features, scaler, encoders, feature_names
            )

            result = risk_scorer.predict_risk(features)

            # Add extracted features summary to response
            result["extracted_features"] = {
                "face_present": frame_features["face_present"],
                "no_of_face": frame_features.get("no_of_face", 0),
                "head_pose": frame_features["head_pose"],
                "gaze_direction": frame_features["gaze_direction"],
                "gaze_on_script": frame_features["gaze_on_script"],
                "hand_count": frame_features["hand_count"],
                "hand_obj_interaction": frame_features.get("hand_obj_interaction", 0),
            }

            # Add enhanced eye metrics
            result["eye_metrics"] = {
                "ear_left": frame_features.get("ear_left", 0),
                "ear_right": frame_features.get("ear_right", 0),
                "ear_avg": frame_features.get("ear_avg", 0),
                "eye_open_ratio": frame_features.get("eye_open_ratio", 0),
                "iris_ratio_left": frame_features.get("iris_ratio_left", 0.5),
                "iris_ratio_right": frame_features.get("iris_ratio_right", 0.5),
                "iris_ratio_avg": frame_features.get("iris_ratio_avg", 0.5),
                "gaze_velocity": frame_features.get("gaze_velocity", 0),
            }

            # Add behavioral warnings
            result["warnings"] = warnings
            if warnings:
                print(f"DEBUG WARNINGS: {[w['message'] for w in warnings]}", flush=True)

            # Add warning summary
            if behavior_analyzer:
                result["warning_summary"] = behavior_analyzer.get_warning_summary()

            # Add gaze tracker window stats
            if gaze_tracker:
                result["gaze_stats"] = gaze_tracker.get_window_stats()

            return jsonify(result)

        # Option 3: Raw named features (dict)
        elif "raw_features" in data:
            frame_features = data["raw_features"]
            features = prepare_realtime_input(
                frame_features, scaler, encoders, feature_names
            )
            result = risk_scorer.predict_risk(features)
            result["warnings"] = []
            result["eye_metrics"] = {}
            return jsonify(result)

        else:
            return jsonify({
                "error": "Provide 'features', 'frame', or 'raw_features'"
            }), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint.

    Accepts JSON with:
      - "features_batch": list of feature arrays

    Returns:
        JSON with list of predictions
    """
    if risk_scorer is None:
        return jsonify({"error": "Model not loaded. Run training first."}), 503

    data = request.get_json()
    if not data or "features_batch" not in data:
        return jsonify({"error": "Provide 'features_batch' as list of feature arrays"}), 400

    try:
        results = []
        for features in data["features_batch"]:
            features_arr = np.array(features, dtype=np.float32).reshape(1, -1)
            result = risk_scorer.predict_risk(features_arr)
            results.append(result)

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/flags", methods=["GET"])
def get_flags():
    """Return all flagged suspicious events for the current session."""
    if behavior_analyzer is None:
        return jsonify({"flags": [], "summary": {}})

    return jsonify({
        "flags": behavior_analyzer.get_all_flags(),
        "summary": behavior_analyzer.get_warning_summary(),
    })


@app.route("/flags/clear", methods=["POST"])
def clear_flags():
    """Clear the flagged events log."""
    if behavior_analyzer:
        behavior_analyzer.clear_flags()
    return jsonify({"status": "cleared"})


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_models()
    print("\n[API] Starting server on http://localhost:5000")
    print("[API] Frontend: http://localhost:5000/")
    print("[API] Health: http://localhost:5000/health")
    print("[API] Flags: http://localhost:5000/flags")
    app.run(host="0.0.0.0", port=5000, debug=False)

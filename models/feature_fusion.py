"""
Feature Fusion Module
=====================
Combines visual features (from webcam/dataset) and behavioral features
into a unified feature vector for model input.
"""

import numpy as np
import pandas as pd
import joblib
import os


def fuse_features(visual_features: np.ndarray, behavioral_features: np.ndarray) -> np.ndarray:
    """
    Concatenate visual and behavioral feature arrays.

    Args:
        visual_features: Array of shape (n_samples, n_visual)
        behavioral_features: Array of shape (n_samples, n_behavioral)

    Returns:
        Combined array of shape (n_samples, n_visual + n_behavioral)
    """
    if visual_features.ndim == 1:
        visual_features = visual_features.reshape(1, -1)
    if behavioral_features.ndim == 1:
        behavioral_features = behavioral_features.reshape(1, -1)

    return np.concatenate([visual_features, behavioral_features], axis=1)


def prepare_realtime_input(
    frame_features: dict,
    scaler,
    encoders: dict,
    feature_names: list,
) -> np.ndarray:
    """
    Convert a single webcam frame's feature dict into a model-ready
    normalized numpy array.

    Args:
        frame_features: Dict from VisualExtractor.extract_features()
        scaler: Fitted MinMaxScaler from training
        encoders: Fitted encoder mapping from training
        feature_names: Ordered list of feature names from training

    Returns:
        Normalized feature array of shape (1, n_features)
    """
    # Build a DataFrame from the single feature dict
    df = pd.DataFrame([frame_features])

    # Compute engineered features for real-time inference
    if "gazePoint_x" in df.columns and "gazePoint_y" in df.columns:
        df["gaze_deviation"] = np.sqrt((df["gazePoint_x"] - 320) ** 2 + (df["gazePoint_y"] - 240) ** 2)
    if "pupil_left_x" in df.columns and "pupil_right_x" in df.columns:
        df["eye_distance"] = np.sqrt((df["pupil_left_x"] - df["pupil_right_x"]) ** 2 + (df["pupil_left_y"] - df["pupil_right_y"]) ** 2)
    if "face_w" in df.columns and "face_h" in df.columns:
        df["face_area"] = df["face_w"] * df["face_h"]
    if "head_pitch" in df.columns and "head_yaw" in df.columns and "head_roll" in df.columns:
        df["head_movement_magnitude"] = np.sqrt(df["head_pitch"] ** 2 + df["head_yaw"] ** 2 + df["head_roll"] ** 2)
    if "hand_count" in df.columns and "hand_obj_interaction" in df.columns:
        df["hand_risk_score"] = df["hand_count"] + df["hand_obj_interaction"] + df.get("phone_present", 0) * 2
    if "left_eye_x" in df.columns and "right_eye_x" in df.columns and "mouth_x" in df.columns:
        eye_mid_x = (df["left_eye_x"] + df["right_eye_x"]) / 2
        eye_mid_y = (df["left_eye_y"] + df["right_eye_y"]) / 2
        df["eye_mouth_distance"] = np.sqrt((eye_mid_x - df["mouth_x"]) ** 2 + (eye_mid_y - df["mouth_y"]) ** 2)
        df["nose_eye_offset"] = np.abs(df.get("nose_tip_x", 0) - eye_mid_x)

    # One-hot encode categorical columns
    categorical_cols = ["head_pose", "gaze_direction"]
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Ensure all expected features exist (add missing as 0)
    for fname in feature_names:
        if fname not in df.columns:
            df[fname] = 0

    # Remove any extra columns and reorder
    df = df.loc[:, ~df.columns.duplicated()] # Remove duplicates if any!
    df = df[feature_names]
    
    print(f"DEBUG: df shape before scaler: {df.shape}")

    # Normalize using the fitted scaler
    X = df.values.astype(np.float32)
    X_normalized = scaler.transform(X)

    return X_normalized


def load_inference_artifacts(saved_models_dir: str) -> tuple:
    """
    Load preprocessing artifacts needed for real-time inference.

    Returns:
        (scaler, encoders, feature_names)
    """
    scaler = joblib.load(os.path.join(saved_models_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(saved_models_dir, "encoders.pkl"))
    feature_names = joblib.load(os.path.join(saved_models_dir, "feature_names.pkl"))
    return scaler, encoders, feature_names

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
    df = df[feature_names]

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

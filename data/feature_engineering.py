"""
Feature Engineering Module
==========================
Derives higher-level behavioral and visual features from raw dataset columns.
These engineered features capture patterns more indicative of suspicious behavior.
"""

import numpy as np
import pandas as pd


# ─── Engineered Feature Functions ─────────────────────────────────────

def compute_gaze_deviation(df: pd.DataFrame) -> pd.Series:
    """
    Calculate gaze deviation from screen center.
    Uses gazePoint_x and gazePoint_y — deviation = Euclidean distance
    from an assumed center point (screen center approximation).
    """
    # Approximate screen center (normalized); use dataset mean as proxy
    cx = df["gazePoint_x"].median()
    cy = df["gazePoint_y"].median()
    deviation = np.sqrt((df["gazePoint_x"] - cx) ** 2 + (df["gazePoint_y"] - cy) ** 2)
    return deviation


def compute_eye_distance(df: pd.DataFrame) -> pd.Series:
    """
    Inter-pupil distance — large changes may indicate head movement or
    the student looking sideways.
    """
    dist = np.sqrt(
        (df["pupil_left_x"] - df["pupil_right_x"]) ** 2
        + (df["pupil_left_y"] - df["pupil_right_y"]) ** 2
    )
    return dist


def compute_face_area(df: pd.DataFrame) -> pd.Series:
    """
    Face bounding-box area — sudden changes indicate the student
    moving closer/farther from the camera.
    """
    return df["face_w"] * df["face_h"]


def compute_head_movement_magnitude(df: pd.DataFrame) -> pd.Series:
    """
    Combined head movement magnitude from pitch, yaw, and roll.
    Higher values = more head movement = potentially suspicious.
    """
    return np.sqrt(
        df["head_pitch"] ** 2 + df["head_yaw"] ** 2 + df["head_roll"] ** 2
    )


def compute_hand_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite hand/phone risk indicator:
      hand_count + hand_obj_interaction + phone_present × 2
    Phone presence is weighted higher as it's a strong cheating signal.
    """
    return (
        df["hand_count"]
        + df["hand_obj_interaction"]
        + df["phone_present"] * 2
    )


def compute_eye_mouth_distance(df: pd.DataFrame) -> pd.Series:
    """
    Distance from eye midpoint to mouth — captures face orientation
    changes that may not show up in head_pose alone.
    """
    eye_mid_x = (df["left_eye_x"] + df["right_eye_x"]) / 2
    eye_mid_y = (df["left_eye_y"] + df["right_eye_y"]) / 2
    return np.sqrt(
        (eye_mid_x - df["mouth_x"]) ** 2
        + (eye_mid_y - df["mouth_y"]) ** 2
    )


def compute_nose_eye_offset(df: pd.DataFrame) -> pd.Series:
    """
    Horizontal offset of nose tip from eye midpoint — indicates
    lateral face rotation not fully captured by head_yaw.
    """
    eye_mid_x = (df["left_eye_x"] + df["right_eye_x"]) / 2
    return np.abs(df["nose_tip_x"] - eye_mid_x)


# ─── Main Feature Engineering Pipeline ────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived features to the dataframe.

    New columns added:
      - gaze_deviation
      - eye_distance
      - face_area
      - head_movement_magnitude
      - hand_risk_score
      - eye_mouth_distance
      - nose_eye_offset

    Returns:
        DataFrame with original + engineered features
    """
    df = df.copy()

    df["gaze_deviation"] = compute_gaze_deviation(df)
    df["eye_distance"] = compute_eye_distance(df)
    df["face_area"] = compute_face_area(df)
    df["head_movement_magnitude"] = compute_head_movement_magnitude(df)
    df["hand_risk_score"] = compute_hand_risk_score(df)
    df["eye_mouth_distance"] = compute_eye_mouth_distance(df)
    df["nose_eye_offset"] = compute_nose_eye_offset(df)

    print(f"[INFO] Added 7 engineered features. New shape: {df.shape}")
    return df


def get_aggregated_features(df: pd.DataFrame) -> dict:
    """
    Compute session-level aggregated statistics for the baseline model.
    Useful when you want a single feature vector per session/window.

    Returns:
        Dict of aggregated statistics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    agg = {}

    for col in numeric_df.columns:
        agg[f"{col}_mean"] = numeric_df[col].mean()
        agg[f"{col}_std"] = numeric_df[col].std()
        agg[f"{col}_min"] = numeric_df[col].min()
        agg[f"{col}_max"] = numeric_df[col].max()

    # Specific behavioral aggregates
    if "gaze_on_script" in df.columns:
        agg["off_screen_ratio"] = 1.0 - df["gaze_on_script"].mean()

    if "phone_present" in df.columns:
        agg["phone_detection_rate"] = df["phone_present"].mean()

    if "hand_obj_interaction" in df.columns:
        agg["hand_interaction_rate"] = df["hand_obj_interaction"].mean()

    return agg


# ─── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.preprocessing import load_raw_data, handle_missing_values

    df = load_raw_data()
    df = handle_missing_values(df)
    df = add_engineered_features(df)

    print(f"\nDataset shape after engineering: {df.shape}")
    print(f"New columns: {df.columns[-7:].tolist()}")
    print(f"\nSample aggregated stats:")
    agg = get_aggregated_features(df.head(50))
    for k, v in list(agg.items())[:10]:
        print(f"  {k}: {v:.4f}")

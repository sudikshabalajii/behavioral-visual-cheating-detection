"""
Data Preprocessing Module
=========================
Handles loading, cleaning, encoding, and normalizing the behavioral
dataset for the multimodal exam cheating detection system.

Dataset: "Students suspicious behaviors detection dataset_V1"
  - 5,500 samples × 38 columns
  - Categorical: head_pose, gaze_direction (with NaN values)
  - Label: binary (0 = Normal, 1 = Suspicious)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# ─── Constants ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "Students suspicious behaviors detection dataset_V1.csv.xls",
)
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Categorical columns that need encoding
CATEGORICAL_COLS = ["head_pose", "gaze_direction"]

# Columns to exclude from features (only the label)
LABEL_COL = "label"

# ─── Data Loading ─────────────────────────────────────────────────────

def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - Categorical columns → 'unknown'
      - Numeric columns → column median
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Fill any remaining numeric NaN with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"[INFO] Missing values handled. Remaining NaN: {df.isnull().sum().sum()}")
    return df


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict = None,
) -> tuple:
    """
    One-hot encode categorical columns (head_pose, gaze_direction).

    Args:
        df: DataFrame with categorical columns
        fit: If True, fit new encoders; otherwise use provided ones
        encoders: Pre-fitted encoder mapping (used during inference)

    Returns:
        (encoded_df, encoders_dict)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        if fit:
            # Get dummies and remember the columns
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            encoders[col] = list(dummies.columns)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            # Apply same encoding as training
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            # Ensure all expected columns exist
            for expected_col in encoders.get(col, []):
                if expected_col not in dummies.columns:
                    dummies[expected_col] = 0
            # Keep only expected columns in correct order
            dummies = dummies[[c for c in encoders[col] if c in dummies.columns]]
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df, encoders


def normalize_features(
    X: np.ndarray,
    fit: bool = True,
    scaler: MinMaxScaler = None,
) -> tuple:
    """
    MinMax-normalize numeric features to [0, 1] range.

    Returns:
        (X_normalized, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()

    if fit:
        X_normalized = scaler.fit_transform(X)
    else:
        X_normalized = scaler.transform(X)

    return X_normalized, scaler


# ─── Main Pipeline ────────────────────────────────────────────────────

def get_processed_data(
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> tuple:
    """
    Full preprocessing pipeline:
      1. Load raw data
      2. Handle missing values
      3. One-hot encode categoricals
      4. Separate features / label
      5. Normalize features (MinMax)
      6. Train/test split (stratified)
      7. Optionally save scaler + encoders for inference

    Returns:
        (X_train, X_test, y_train, y_test, feature_names, scaler, encoders)
    """
    # Step 1-2: Load and clean
    df = load_raw_data()
    df = handle_missing_values(df)

    # Step 3: Encode categoricals
    df, encoders = encode_categoricals(df, fit=True)

    # Step 4: Separate features and label
    y = df[LABEL_COL].values
    X_df = df.drop(columns=[LABEL_COL])
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)

    # Step 5: Normalize
    X_normalized, scaler = normalize_features(X, fit=True)

    # Step 6: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[INFO] Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"[INFO] Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[INFO] Test label distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Step 7: Save artifacts for inference
    if save_artifacts:
        joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, "scaler.pkl"))
        joblib.dump(encoders, os.path.join(SAVED_MODELS_DIR, "encoders.pkl"))
        joblib.dump(feature_names, os.path.join(SAVED_MODELS_DIR, "feature_names.pkl"))
        print(f"[INFO] Saved preprocessing artifacts to {SAVED_MODELS_DIR}")

    return X_train, X_test, y_train, y_test, feature_names, scaler, encoders


# ─── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, names, scaler, enc = get_processed_data()
    print(f"\nFeature count: {len(names)}")
    print(f"Features: {names}")

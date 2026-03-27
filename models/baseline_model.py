"""
Baseline Models (Model B — Random Forest & XGBoost)
====================================================
Traditional ML baselines using aggregated/flattened features.
Provides a performance reference for comparison with the CNN+LSTM.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] XGBoost not installed. Only Random Forest will be trained.")


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Flattened/aggregated feature array (n_samples, n_features)
        y_train: Labels
        n_estimators: Number of decision trees
        random_state: Reproducibility seed

    Returns:
        Trained RandomForestClassifier
    """
    print("\n" + "=" * 60)
    print("Training Random Forest (Baseline)")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    print(f"  Train accuracy: {train_acc:.4f}")
    return rf


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> object:
    """
    Train an XGBoost classifier.

    Returns:
        Trained XGBClassifier (or None if xgboost not installed)
    """
    if not HAS_XGBOOST:
        print("[SKIP] XGBoost not available")
        return None

    print("\n" + "=" * 60)
    print("Training XGBoost (Baseline)")
    print("=" * 60)

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, xgb.predict(X_train))
    print(f"  Train accuracy: {train_acc:.4f}")
    return xgb


def evaluate_baseline(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Baseline",
) -> dict:
    """
    Evaluate a baseline model and return detailed metrics.

    Returns:
        Dict with accuracy, precision, recall, f1, predictions, probabilities
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


def save_baseline_models(
    rf_model,
    xgb_model,
    save_dir: str,
):
    """Save trained baseline models to disk."""
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(rf_model, os.path.join(save_dir, "random_forest.pkl"))
    print(f"[INFO] Saved Random Forest to {save_dir}")

    if xgb_model is not None:
        joblib.dump(xgb_model, os.path.join(save_dir, "xgboost.pkl"))
        print(f"[INFO] Saved XGBoost to {save_dir}")


# ─── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with dummy data
    X = np.random.rand(200, 47).astype(np.float32)
    y = np.random.randint(0, 2, size=200)

    rf = train_random_forest(X[:160], y[:160])
    rf_metrics = evaluate_baseline(rf, X[160:], y[160:], "Random Forest")
    print(f"\nRF Test accuracy: {rf_metrics['accuracy']:.4f}")

    xgb = train_xgboost(X[:160], y[:160])
    if xgb:
        xgb_metrics = evaluate_baseline(xgb, X[160:], y[160:], "XGBoost")
        print(f"XGB Test accuracy: {xgb_metrics['accuracy']:.4f}")

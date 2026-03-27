"""
Risk Scoring Module
===================
Converts model prediction probabilities into a 0-100 risk score
and classifies into risk tiers: Normal / Moderate / High Risk.
"""

import numpy as np


# ─── Risk Tiers ───────────────────────────────────────────────────────
RISK_TIERS = {
    "Normal": (0, 30),
    "Moderate Risk": (30, 70),
    "High Risk": (70, 100),
}


def compute_risk_score(probability: float) -> int:
    """
    Convert model sigmoid probability [0, 1] to risk score [0, 100].

    Uses a slightly non-linear mapping that emphasizes extreme probabilities:
      score = round(probability^0.85 × 100)

    This makes the score more sensitive near high/low ends while
    keeping moderate probabilities reasonably mapped.

    Args:
        probability: Model output probability (0 = normal, 1 = suspicious)

    Returns:
        Integer risk score 0-100
    """
    probability = float(np.clip(probability, 0.0, 1.0))
    # Non-linear mapping for better discrimination at extremes
    score = probability ** 0.85 * 100
    return int(np.clip(round(score), 0, 100))


def classify_risk(score: int) -> str:
    """
    Classify a risk score into a tier.

    Args:
        score: Risk score 0-100

    Returns:
        Risk level string: "Normal", "Moderate Risk", or "High Risk"
    """
    for tier_name, (low, high) in RISK_TIERS.items():
        if low <= score < high:
            return tier_name
    return "High Risk"  # score == 100


def get_risk_color(score: int) -> str:
    """
    Get the associated color for a risk score (for UI display).

    Returns:
        Hex color string
    """
    if score < 30:
        return "#22c55e"   # Green
    elif score < 70:
        return "#f59e0b"   # Amber
    else:
        return "#ef4444"   # Red


def compute_batch_risk(probabilities: np.ndarray) -> list:
    """
    Compute risk scores for a batch of predictions.

    Args:
        probabilities: Array of model output probabilities

    Returns:
        List of dicts with score, level, and color
    """
    results = []
    for prob in probabilities:
        score = compute_risk_score(prob)
        level = classify_risk(score)
        color = get_risk_color(score)
        results.append({
            "risk_score": score,
            "risk_level": level,
            "risk_color": color,
            "raw_probability": float(prob),
        })
    return results


class RiskScorer:
    """
    End-to-end inference wrapper that combines model prediction
    with risk scoring.
    """

    def __init__(self, model, scaler=None, encoders=None, feature_names=None):
        """
        Args:
            model: Trained Keras or sklearn model
            scaler: Fitted MinMaxScaler
            encoders: Fitted encoder mapping
            feature_names: Ordered feature name list
        """
        self.model = model
        self.scaler = scaler
        self.encoders = encoders
        self.feature_names = feature_names

    def predict_risk(self, features: np.ndarray) -> dict:
        """
        Predict and score risk from processed features.

        Args:
            features: Model-ready feature array

        Returns:
            Dict with prediction, risk_score, risk_level
        """
        # Get probability
        if hasattr(self.model, "predict_proba"):
            # sklearn model
            prob = self.model.predict_proba(features)[:, 1][0]
        else:
            # Keras model
            prob = float(self.model.predict(features, verbose=0).flatten()[0])

        prediction = 1 if prob >= 0.5 else 0
        score = compute_risk_score(prob)
        level = classify_risk(score)
        color = get_risk_color(score)

        return {
            "prediction": prediction,
            "prediction_label": "Suspicious" if prediction == 1 else "Normal",
            "probability": round(float(prob), 4),
            "risk_score": score,
            "risk_level": level,
            "risk_color": color,
        }


# ─── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    test_probs = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
    print("Probability → Risk Score → Level")
    print("-" * 50)
    for p in test_probs:
        score = compute_risk_score(p)
        level = classify_risk(score)
        color = get_risk_color(score)
        print(f"  {p:.2f} → {score:3d} → {level:<15s} {color}")

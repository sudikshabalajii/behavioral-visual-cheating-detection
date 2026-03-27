"""
Training Pipeline
=================
Orchestrates the complete training workflow:
  1. Load and preprocess data
  2. Engineer features
  3. Build sequences for LSTM
  4. Train CNN+LSTM (Model A) and RF/XGBoost (Model B)
  5. Evaluate and compare all models
  6. Generate plots (training curves, confusion matrices)
  7. Save models and artifacts
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.preprocessing import get_processed_data, load_raw_data, handle_missing_values, encode_categoricals, normalize_features, LABEL_COL
from data.feature_engineering import add_engineered_features
from models.sequence_builder import build_sequences
from models.lstm_model import build_cnn_lstm_model, train_lstm_model, evaluate_lstm_model
from models.baseline_model import (
    train_random_forest, train_xgboost,
    evaluate_baseline, save_baseline_models,
)

# ─── Configuration ────────────────────────────────────────────────────
WINDOW_SIZE = 10       # Sequence length for LSTM
EPOCHS = 50            # Max training epochs (early stopping enabled)
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_training_history(history: dict, save_path: str):
    """Plot training vs validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN+LSTM Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history["accuracy"], label="Train", linewidth=2)
    axes[0].plot(history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history["loss"], label="Train", linewidth=2)
    axes[1].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training history plot saved: {save_path}")


def plot_confusion_matrices(results: list, y_test: np.ndarray, save_path: str):
    """Plot confusion matrices for all models side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    labels = ["Normal", "Suspicious"]

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test[:len(res["y_pred"])], res["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_title(f"{res['model_name']}\nAccuracy: {res['accuracy']:.4f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix plot saved: {save_path}")


def plot_model_comparison(results: list, save_path: str):
    """Bar chart comparing all models across metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = [r["model_name"] for r in results]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, res in enumerate(results):
        values = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=res["model_name"])
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Comparison plot saved: {save_path}")


def print_comparison_table(results: list):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("-" * 70)

    for res in results:
        row = (
            f"{res['model_name']:<20} "
            f"{res['accuracy']:>10.4f} "
            f"{res['precision']:>10.4f} "
            f"{res['recall']:>10.4f} "
            f"{res['f1']:>10.4f}"
        )
        print(row)
    print("=" * 70)


# ─── Main Training Pipeline ──────────────────────────────────────────

def main():
    print("=" * 70)
    print("MULTIMODAL EXAM CHEATING DETECTION — TRAINING PIPELINE")
    print("=" * 70)

    # ── Step 1: Load and preprocess ───────────────────────────────────
    print("\n[PHASE 1] Loading and preprocessing data...")
    df = load_raw_data()
    df = handle_missing_values(df)

    # ── Step 2: Feature engineering ───────────────────────────────────
    print("\n[PHASE 2] Engineering features...")
    df = add_engineered_features(df)

    # ── Step 3: Encode and normalize ──────────────────────────────────
    print("\n[PHASE 3] Encoding and normalizing...")
    y = df[LABEL_COL].values
    df_features = df.drop(columns=[LABEL_COL])
    df_encoded, encoders = encode_categoricals(df_features, fit=True)

    feature_names = list(df_encoded.columns)
    X = df_encoded.values.astype(np.float32)
    X_normalized, scaler = normalize_features(X, fit=True)

    # Save artifacts
    import joblib
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(SAVED_MODELS_DIR, "encoders.pkl"))
    joblib.dump(feature_names, os.path.join(SAVED_MODELS_DIR, "feature_names.pkl"))

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    all_results = []

    # ── Step 4: Baseline Models (Model B) ─────────────────────────────
    print("\n[PHASE 4] Training baseline models...")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_baseline(rf_model, X_test, y_test, "Random Forest")
    all_results.append(rf_results)

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    if xgb_model:
        xgb_results = evaluate_baseline(xgb_model, X_test, y_test, "XGBoost")
        all_results.append(xgb_results)

    save_baseline_models(rf_model, xgb_model, SAVED_MODELS_DIR)

    # ── Step 5: CNN+LSTM (Model A) ────────────────────────────────────
    print("\n[PHASE 5] Building sequences and training CNN+LSTM...")

    # Build sequences
    X_train_seq, y_train_seq = build_sequences(X_train, y_train, WINDOW_SIZE)
    X_test_seq, y_test_seq = build_sequences(X_test, y_test, WINDOW_SIZE)

    # Build model
    input_shape = (WINDOW_SIZE, X_train.shape[1])
    lstm_model = build_cnn_lstm_model(input_shape)
    lstm_model.summary()

    # Train
    model_save_path = os.path.join(SAVED_MODELS_DIR, "cnn_lstm_model.keras")
    history = train_lstm_model(
        lstm_model,
        X_train_seq, y_train_seq,
        X_test_seq, y_test_seq,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_path=model_save_path,
    )

    # Evaluate
    lstm_eval = evaluate_lstm_model(lstm_model, X_test_seq, y_test_seq)

    lstm_results = {
        "model_name": "CNN+LSTM",
        "accuracy": lstm_eval["accuracy"],
        "precision": precision_score(y_test_seq, lstm_eval["y_pred"], zero_division=0),
        "recall": recall_score(y_test_seq, lstm_eval["y_pred"], zero_division=0),
        "f1": f1_score(y_test_seq, lstm_eval["y_pred"], zero_division=0),
        "y_pred": lstm_eval["y_pred"],
        "y_prob": lstm_eval["y_prob"],
    }
    all_results.append(lstm_results)

    # ── Step 6: Generate Plots ────────────────────────────────────────
    print("\n[PHASE 6] Generating plots...")

    # Training curves
    plot_training_history(history, os.path.join(PLOTS_DIR, "training_history.png"))

    # Confusion matrices (use appropriate y_test for each)
    # For baselines, use full y_test; for LSTM, use y_test_seq
    baseline_results = [r for r in all_results if r["model_name"] != "CNN+LSTM"]
    if baseline_results:
        plot_confusion_matrices(baseline_results, y_test,
                                os.path.join(PLOTS_DIR, "confusion_baseline.png"))

    lstm_results_list = [r for r in all_results if r["model_name"] == "CNN+LSTM"]
    if lstm_results_list:
        plot_confusion_matrices(lstm_results_list, y_test_seq,
                                os.path.join(PLOTS_DIR, "confusion_lstm.png"))

    # Model comparison bar chart
    plot_model_comparison(all_results, os.path.join(PLOTS_DIR, "model_comparison.png"))

    # ── Step 7: Print Results ─────────────────────────────────────────
    print_comparison_table(all_results)

    # Per-model classification reports
    for res in all_results:
        print(f"\n{'-' * 40}")
        print(f"Classification Report — {res['model_name']}")
        print("-" * 40)
        if res["model_name"] == "CNN+LSTM":
            print(classification_report(y_test_seq, res["y_pred"],
                                        target_names=["Normal", "Suspicious"]))
        else:
            print(classification_report(y_test, res["y_pred"],
                                        target_names=["Normal", "Suspicious"]))

    # Save window size for inference
    joblib.dump(WINDOW_SIZE, os.path.join(SAVED_MODELS_DIR, "window_size.pkl"))

    print("\n[DONE] All models trained and saved to:", SAVED_MODELS_DIR)
    print("[DONE] Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()

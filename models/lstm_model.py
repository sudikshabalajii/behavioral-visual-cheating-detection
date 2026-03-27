"""
CNN + LSTM Model (Model A — Primary Deep Learning)
===================================================
A hybrid Conv1D + LSTM architecture for temporal sequence classification.
Processes sequences of multimodal features (visual + behavioral) to
detect suspicious exam behavior.

Architecture:
  Input → Conv1D(64) → Conv1D(128) → LSTM(64) → Dense(32) → Dense(1, sigmoid)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


def build_cnn_lstm_model(input_shape: tuple, learning_rate: float = 0.001):
    """
    Build the CNN+LSTM hybrid model.

    Args:
        input_shape: (window_size, n_features) — e.g., (10, 47)
        learning_rate: Adam optimizer learning rate

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # ── Temporal Convolution Block ────────────────────────────────
        # Extract local patterns in the feature time-series
        layers.Conv1D(
            filters=64, kernel_size=3, activation="relu",
            padding="same", input_shape=input_shape,
            name="conv1d_1",
        ),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.25, name="dropout_1"),

        layers.Conv1D(
            filters=128, kernel_size=3, activation="relu",
            padding="same", name="conv1d_2",
        ),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.25, name="dropout_2"),

        # ── Recurrent Block ───────────────────────────────────────────
        # Capture long-term temporal dependencies
        layers.LSTM(64, return_sequences=False, name="lstm"),
        layers.Dropout(0.3, name="dropout_3"),

        # ── Classification Head ───────────────────────────────────────
        layers.Dense(32, activation="relu", name="dense_1"),
        layers.Dropout(0.3, name="dropout_4"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_lstm_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    save_path: str = None,
) -> dict:
    """
    Train the CNN+LSTM model with early stopping and learning rate reduction.

    Args:
        model: Compiled Keras model
        X_train: Training sequences (n, window_size, features)
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        epochs: Maximum training epochs
        batch_size: Batch size
        save_path: Path to save best model weights

    Returns:
        Training history dict
    """
    callback_list = [
        # Stop early if validation loss doesn't improve for 10 epochs
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Save best model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        callback_list.append(
            callbacks.ModelCheckpoint(
                save_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            )
        )

    print("\n" + "=" * 60)
    print("Training CNN+LSTM Model (Model A)")
    print("=" * 60)
    print(f"  Input shape: {X_train.shape}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1,
    )

    return history.history


def evaluate_lstm_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the trained model and return metrics.

    Returns:
        Dict with loss, accuracy, predictions, and probabilities
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ─── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test model building with dummy data
    window_size, n_features = 10, 47
    model = build_cnn_lstm_model((window_size, n_features))
    model.summary()

    # Quick train on random data
    X = np.random.rand(200, window_size, n_features).astype(np.float32)
    y = np.random.randint(0, 2, size=200)
    history = train_lstm_model(model, X[:160], y[:160], X[160:], y[160:], epochs=5)
    print(f"\nFinal val_accuracy: {history['val_accuracy'][-1]:.4f}")

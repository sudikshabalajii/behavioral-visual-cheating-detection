"""
Sequence Builder Module
=======================
Converts flat feature data into temporal sequences for LSTM/CNN+LSTM
models using a sliding window approach.
"""

import numpy as np


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 10,
    step_size: int = 1,
) -> tuple:
    """
    Create sliding-window sequences from feature and label arrays.

    Each sequence is window_size consecutive feature vectors, and
    the label is the last label in the window (majority voting
    could also be used).

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        window_size: Number of time steps per sequence
        step_size: Step between consecutive windows

    Returns:
        (X_sequences, y_sequences) where:
          X_sequences shape: (n_sequences, window_size, n_features)
          y_sequences shape: (n_sequences,)

    Example:
        Given 100 samples with window_size=10, step=1:
        → 91 sequences, each (10, n_features)
    """
    if len(X) < window_size:
        raise ValueError(
            f"Not enough samples ({len(X)}) for window_size={window_size}"
        )

    X_seq = []
    y_seq = []

    for i in range(0, len(X) - window_size + 1, step_size):
        X_seq.append(X[i : i + window_size])
        # Use the last label in the window as the sequence label
        y_seq.append(y[i + window_size - 1])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int32)

    print(f"[INFO] Built {len(X_seq)} sequences of shape {X_seq.shape[1:]}")
    return X_seq, y_seq


def build_sequences_realtime(
    feature_buffer: list,
    window_size: int = 10,
) -> np.ndarray:
    """
    Build a single sequence from a rolling buffer of live features.
    Used during real-time inference.

    Args:
        feature_buffer: List of feature arrays (each shape: (n_features,))
        window_size: Expected sequence length

    Returns:
        Array of shape (1, window_size, n_features)
    """
    if len(feature_buffer) < window_size:
        # Pad with zeros if not enough frames yet
        n_features = len(feature_buffer[0]) if feature_buffer else 1
        padding = [np.zeros(n_features)] * (window_size - len(feature_buffer))
        feature_buffer = padding + list(feature_buffer)

    # Take the last window_size frames
    recent = feature_buffer[-window_size:]
    sequence = np.array(recent, dtype=np.float32)
    return sequence.reshape(1, window_size, -1)


# ─── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with dummy data
    X_dummy = np.random.rand(100, 20).astype(np.float32)
    y_dummy = np.random.randint(0, 2, size=100)

    X_seq, y_seq = build_sequences(X_dummy, y_dummy, window_size=10, step_size=1)
    print(f"Input shape: {X_dummy.shape}")
    print(f"Sequence shape: {X_seq.shape}")
    print(f"Labels shape: {y_seq.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(y_seq, return_counts=True)))}")

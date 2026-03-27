"""
Feature Utilities (Webcam)
==========================
Rolling-window statistics tracker for real-time webcam features.
Computes temporal behavioral signals like gaze deviation frequency,
head turn frequency, and off-screen gaze duration.
"""

import time
from collections import deque
import numpy as np


class GazeTracker:
    """
    Tracks webcam-extracted features over a rolling window and
    computes temporal statistics for risk assessment.
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of frames to track (at ~15fps → 2 seconds)
        """
        self.window_size = window_size
        self.frame_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # Counters for session-level stats
        self.total_frames = 0
        self.gaze_off_screen_count = 0
        self.head_turn_count = 0
        self.last_head_pose = "forward"
        self.session_start_time = time.time()

    def update(self, features: dict):
        """
        Add a new frame's features to the rolling window.

        Args:
            features: Dict from VisualExtractor.extract_features()
        """
        self.frame_history.append(features)
        self.timestamps.append(time.time())
        self.total_frames += 1

        # Track gaze off-screen events
        if features.get("gaze_on_script", 1) == 0 and features.get("face_present", 0):
            self.gaze_off_screen_count += 1

        # Track head turns (pose changes)
        current_pose = features.get("head_pose", "forward")
        if current_pose != self.last_head_pose and current_pose != "unknown":
            self.head_turn_count += 1
        self.last_head_pose = current_pose

    def get_window_stats(self) -> dict:
        """
        Compute statistics over the current rolling window.

        Returns:
            Dict with temporal features:
              - gaze_deviation_count: frames with gaze off-center in window
              - head_turn_frequency: head pose changes per second
              - off_screen_duration: fraction of window with gaze off-screen
              - avg_head_movement: mean head movement magnitude in window
              - face_absence_rate: fraction of window without face detection
              - hand_presence_rate: fraction of window with hands visible
        """
        if len(self.frame_history) == 0:
            return self._get_empty_stats()

        frames = list(self.frame_history)
        n = len(frames)

        # Time span of window
        if len(self.timestamps) >= 2:
            time_span = self.timestamps[-1] - self.timestamps[0]
        else:
            time_span = 1.0  # Avoid division by zero

        time_span = max(time_span, 0.001)

        # Gaze deviation count (not on script)
        gaze_off = sum(1 for f in frames
                       if f.get("gaze_on_script", 1) == 0 and f.get("face_present", 0))

        # Head turn frequency (pose changes per second)
        pose_changes = 0
        for i in range(1, n):
            prev = frames[i - 1].get("head_pose", "forward")
            curr = frames[i].get("head_pose", "forward")
            if prev != curr and curr != "unknown":
                pose_changes += 1

        # Average head movement magnitude
        movements = []
        for f in frames:
            mag = np.sqrt(
                f.get("head_pitch", 0) ** 2
                + f.get("head_yaw", 0) ** 2
                + f.get("head_roll", 0) ** 2
            )
            movements.append(mag)

        # Face absence rate
        face_absent = sum(1 for f in frames if f.get("face_present", 0) == 0)

        # Hand presence rate
        hand_present = sum(1 for f in frames if f.get("hand_count", 0) > 0)

        return {
            "gaze_deviation_count": gaze_off,
            "head_turn_frequency": pose_changes / time_span,
            "off_screen_duration": gaze_off / n,
            "avg_head_movement": float(np.mean(movements)),
            "max_head_movement": float(np.max(movements)) if movements else 0.0,
            "face_absence_rate": face_absent / n,
            "hand_presence_rate": hand_present / n,
        }

    def get_session_stats(self) -> dict:
        """
        Compute accumulated session-level statistics.

        Returns:
            Dict with session-wide stats
        """
        elapsed = max(time.time() - self.session_start_time, 0.001)

        return {
            "session_duration_sec": elapsed,
            "total_frames_processed": self.total_frames,
            "total_gaze_off_screen": self.gaze_off_screen_count,
            "gaze_off_screen_rate": self.gaze_off_screen_count / max(self.total_frames, 1),
            "total_head_turns": self.head_turn_count,
            "head_turns_per_minute": (self.head_turn_count / elapsed) * 60,
        }

    def get_summary_stats(self) -> dict:
        """
        Combined window + session statistics for display overlay.
        """
        stats = self.get_window_stats()
        stats.update(self.get_session_stats())
        return stats

    def _get_empty_stats(self) -> dict:
        """Return zeroed stats when no data is available."""
        return {
            "gaze_deviation_count": 0,
            "head_turn_frequency": 0.0,
            "off_screen_duration": 0.0,
            "avg_head_movement": 0.0,
            "max_head_movement": 0.0,
            "face_absence_rate": 0.0,
            "hand_presence_rate": 0.0,
        }

    def reset(self):
        """Reset all tracking data."""
        self.frame_history.clear()
        self.timestamps.clear()
        self.total_frames = 0
        self.gaze_off_screen_count = 0
        self.head_turn_count = 0
        self.last_head_pose = "forward"
        self.session_start_time = time.time()

"""
Behavioral Analysis Engine
===========================
Detects suspicious behavior patterns from webcam features in real-time.
Classifies events into severity tiers and produces actionable warnings.

Suspicious behaviors detected:
  1. Rapid eye movement (frequent gaze shifts)
  2. Prolonged gaze away from screen
  3. Face absence (no face detected)
  4. Multiple faces in frame
  5. Excessive head turns
  6. Prolonged eye closure
  7. Hand near face (potential phone / whispering)
"""

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
from typing import List, Optional


# ─── Enums ────────────────────────────────────────────────────────────

class EventType(str, Enum):
    GAZE_AWAY = "gaze_away"
    RAPID_EYE = "rapid_eye_movement"
    FACE_ABSENT = "face_absent"
    MULTI_FACE = "multiple_faces"
    HEAD_TURN = "excessive_head_turns"
    EYE_CLOSED = "eye_closure"
    HAND_FACE = "hand_near_face"
    TAB_SWITCH = "tab_switch"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ─── Data Classes ─────────────────────────────────────────────────────

@dataclass
class SuspiciousEvent:
    """A single detected suspicious behavior event."""
    event_type: str
    severity: str
    message: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "time_str": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "confidence": round(self.confidence, 2),
        }


# ─── Thresholds Configuration ────────────────────────────────────────

@dataclass
class BehaviorThresholds:
    """Configurable thresholds for behavior detection."""

    # Gaze away: seconds of continuous off-screen gaze to trigger
    gaze_away_warning_sec: float = 3.0
    gaze_away_critical_sec: float = 8.0

    # Rapid eye movement: direction changes within window to trigger
    rapid_eye_changes_warning: int = 4
    rapid_eye_changes_critical: int = 7
    rapid_eye_window_sec: float = 3.0

    # Face absence: seconds without face to trigger
    face_absent_warning_sec: float = 3.0
    face_absent_critical_sec: float = 8.0

    # Multiple faces
    multi_face_enabled: bool = True

    # Head turns: count within window to trigger
    head_turn_warning_count: int = 4
    head_turn_critical_count: int = 8
    head_turn_window_sec: float = 10.0

    # Eye closure (EAR-based): seconds of closed eyes to trigger
    ear_closed_threshold: float = 0.18
    eye_closed_warning_sec: float = 2.0
    eye_closed_critical_sec: float = 5.0

    # Hand near face
    hand_face_warning_sec: float = 3.0

    # Cooldown: minimum seconds between same-type warnings
    cooldown_sec: float = 5.0


# ─── Behavior Analyzer ───────────────────────────────────────────────

class BehaviorAnalyzer:
    """
    Analyzes webcam features frame-by-frame and detects suspicious
    behavior patterns. Maintains internal state for temporal analysis.
    """

    def __init__(self, thresholds: Optional[BehaviorThresholds] = None):
        self.thresholds = thresholds or BehaviorThresholds()

        # ── Internal state ────────────────────────────────────────
        # Gaze tracking
        self._gaze_away_start: Optional[float] = None
        self._gaze_directions: deque = deque(maxlen=60)
        self._gaze_timestamps: deque = deque(maxlen=60)

        # Face tracking
        self._face_absent_start: Optional[float] = None

        # Head turn tracking
        self._head_poses: deque = deque(maxlen=60)
        self._head_timestamps: deque = deque(maxlen=60)
        self._last_head_pose: str = "forward"

        # Eye closure tracking
        self._eye_closed_start: Optional[float] = None

        # Hand near face tracking
        self._hand_face_start: Optional[float] = None

        # Cooldowns (event_type -> last_triggered_time)
        self._cooldowns: dict = {}

        # Session flag log
        self.flagged_events: List[SuspiciousEvent] = []
        self.warning_counts: dict = {e.value: 0 for e in EventType}
        self.session_start: float = time.time()

    def analyze_frame(self, features: dict) -> List[dict]:
        """
        Analyze a single frame's features and return any detected
        suspicious events.

        Args:
            features: Dict from VisualExtractor.extract_features()
                      (must include ear_left, ear_right, ear_avg,
                       gaze_velocity if available)

        Returns:
            List of event dicts (may be empty if nothing suspicious)
        """
        now = time.time()
        events: List[SuspiciousEvent] = []

        # ── 1. Gaze Away Detection ────────────────────────────────
        gaze_events = self._check_gaze_away(features, now)
        events.extend(gaze_events)

        # ── 2. Rapid Eye Movement Detection ───────────────────────
        rapid_events = self._check_rapid_eye_movement(features, now)
        events.extend(rapid_events)

        # ── 3. Face Absence Detection ─────────────────────────────
        face_events = self._check_face_absence(features, now)
        events.extend(face_events)

        # ── 4. Multiple Faces Detection ───────────────────────────
        multi_events = self._check_multiple_faces(features, now)
        events.extend(multi_events)

        # ── 5. Excessive Head Turns Detection ─────────────────────
        head_events = self._check_head_turns(features, now)
        events.extend(head_events)

        # ── 6. Eye Closure Detection ──────────────────────────────
        eye_events = self._check_eye_closure(features, now)
        events.extend(eye_events)

        # ── 7. Hand Near Face Detection ───────────────────────────
        hand_events = self._check_hand_near_face(features, now)
        events.extend(hand_events)

        # Store flagged events
        for event in events:
            self.flagged_events.append(event)
            self.warning_counts[event.event_type] = \
                self.warning_counts.get(event.event_type, 0) + 1

        return [e.to_dict() for e in events]

    # ─── Individual Detectors ─────────────────────────────────────

    def _check_gaze_away(self, features: dict, now: float) -> List[SuspiciousEvent]:
        """Detect prolonged gaze away from screen center."""
        events = []
        gaze_on = features.get("gaze_on_script", 1)
        face_present = features.get("face_present", 0)

        if face_present and gaze_on == 0:
            if self._gaze_away_start is None:
                self._gaze_away_start = now

            duration = now - self._gaze_away_start

            if duration >= self.thresholds.gaze_away_critical_sec:
                if self._can_trigger(EventType.GAZE_AWAY, now):
                    direction = features.get("gaze_direction", "unknown")
                    events.append(SuspiciousEvent(
                        event_type=EventType.GAZE_AWAY.value,
                        severity=Severity.CRITICAL.value,
                        message=f"Looking away from screen for {duration:.0f}s "
                                f"(direction: {direction})",
                        confidence=min(0.95, 0.7 + duration * 0.03),
                    ))
            elif duration >= self.thresholds.gaze_away_warning_sec:
                if self._can_trigger(EventType.GAZE_AWAY, now):
                    direction = features.get("gaze_direction", "unknown")
                    events.append(SuspiciousEvent(
                        event_type=EventType.GAZE_AWAY.value,
                        severity=Severity.WARNING.value,
                        message=f"Gaze away from screen for {duration:.0f}s "
                                f"(direction: {direction})",
                        confidence=min(0.85, 0.6 + duration * 0.05),
                    ))
        else:
            self._gaze_away_start = None

        return events

    def _check_rapid_eye_movement(self, features: dict, now: float) -> List[SuspiciousEvent]:
        """Detect rapid gaze direction changes."""
        events = []
        gaze_dir = features.get("gaze_direction", "unknown")

        if gaze_dir != "unknown" and features.get("face_present", 0):
            self._gaze_directions.append(gaze_dir)
            self._gaze_timestamps.append(now)

            # Count direction changes in the window
            window_start = now - self.thresholds.rapid_eye_window_sec
            changes = 0
            prev_dir = None
            for i, (d, t) in enumerate(zip(self._gaze_directions, self._gaze_timestamps)):
                if t >= window_start:
                    if prev_dir is not None and d != prev_dir:
                        changes += 1
                    prev_dir = d

            if changes >= self.thresholds.rapid_eye_changes_critical:
                if self._can_trigger(EventType.RAPID_EYE, now):
                    events.append(SuspiciousEvent(
                        event_type=EventType.RAPID_EYE.value,
                        severity=Severity.CRITICAL.value,
                        message=f"Rapid eye movement detected "
                                f"({changes} shifts in "
                                f"{self.thresholds.rapid_eye_window_sec:.0f}s)",
                        confidence=min(0.95, 0.7 + changes * 0.03),
                    ))
            elif changes >= self.thresholds.rapid_eye_changes_warning:
                if self._can_trigger(EventType.RAPID_EYE, now):
                    events.append(SuspiciousEvent(
                        event_type=EventType.RAPID_EYE.value,
                        severity=Severity.WARNING.value,
                        message=f"Frequent eye movement detected "
                                f"({changes} shifts in "
                                f"{self.thresholds.rapid_eye_window_sec:.0f}s)",
                        confidence=min(0.85, 0.5 + changes * 0.05),
                    ))

        return events

    def _check_face_absence(self, features: dict, now: float) -> List[SuspiciousEvent]:
        """Detect prolonged face absence from frame."""
        events = []
        face_present = features.get("face_present", 0)

        if not face_present:
            if self._face_absent_start is None:
                self._face_absent_start = now

            duration = now - self._face_absent_start

            if duration >= self.thresholds.face_absent_critical_sec:
                if self._can_trigger(EventType.FACE_ABSENT, now):
                    events.append(SuspiciousEvent(
                        event_type=EventType.FACE_ABSENT.value,
                        severity=Severity.CRITICAL.value,
                        message=f"Face not detected for {duration:.0f}s — "
                                f"please remain in front of the camera",
                        confidence=0.95,
                    ))
            elif duration >= self.thresholds.face_absent_warning_sec:
                if self._can_trigger(EventType.FACE_ABSENT, now):
                    events.append(SuspiciousEvent(
                        event_type=EventType.FACE_ABSENT.value,
                        severity=Severity.WARNING.value,
                        message=f"Face not detected for {duration:.0f}s",
                        confidence=0.85,
                    ))
        else:
            self._face_absent_start = None

        return events

    def _check_multiple_faces(self, features: dict, now: float) -> List[SuspiciousEvent]:
        """Detect multiple faces in the frame."""
        events = []
        if not self.thresholds.multi_face_enabled:
            return events

        face_count = features.get("no_of_face", 0)
        if face_count > 1:
            if self._can_trigger(EventType.MULTI_FACE, now):
                events.append(SuspiciousEvent(
                    event_type=EventType.MULTI_FACE.value,
                    severity=Severity.CRITICAL.value,
                    message=f"Multiple faces detected ({face_count} faces) — "
                            f"only the exam taker should be visible",
                    confidence=0.9,
                ))

        return events

    def _check_head_turns(self, features: dict, now: float) -> List[SuspiciousEvent]:
        """Detect excessive head turning."""
        events = []
        head_pose = features.get("head_pose", "unknown")

        if head_pose != "unknown" and features.get("face_present", 0):
            # Track pose changes
            if head_pose != self._last_head_pose:
                self._head_poses.append(head_pose)
                self._head_timestamps.append(now)
            self._last_head_pose = head_pose

            # Count turns in window
            window_start = now - self.thresholds.head_turn_window_sec
            turns = sum(1 for t in self._head_timestamps if t >= window_start)

            if turns >= self.thresholds.head_turn_critical_count:
                if self._can_trigger(EventType.HEAD_TURN, now):
                    events.append(SuspiciousEvent(
                        event_type=EventType.HEAD_TURN.value,
                        severity=Severity.CRITICAL.value,
                        message=f"Excessive head turning detected "
                                f"({turns} turns in "
                                f"{self.thresholds.head_turn_window_sec:.0f}s)",
                        confidence=min(0.95, 0.6 + turns * 0.04),
                    ))
            elif turns >= self.thresholds.head_turn_warning_count:
                if self._can_trigger(EventType.HEAD_TURN, now):
                    events.append(SuspiciousEvent(
                        event_type=EventType.HEAD_TURN.value,
                        severity=Severity.WARNING.value,
                        message=f"Frequent head turning detected "
                                f"({turns} turns in "
                                f"{self.thresholds.head_turn_window_sec:.0f}s)",
                        confidence=min(0.85, 0.5 + turns * 0.05),
                    ))

        return events

    def _check_eye_closure(self, features: dict, now: float) -> List[SuspiciousEvent]:
        """Detect prolonged eye closure using Eye Aspect Ratio."""
        events = []
        ear_avg = features.get("ear_avg", None)

        if ear_avg is None or not features.get("face_present", 0):
            self._eye_closed_start = None
            return events

        if ear_avg < self.thresholds.ear_closed_threshold:
            if self._eye_closed_start is None:
                self._eye_closed_start = now

            duration = now - self._eye_closed_start

            if duration >= self.thresholds.eye_closed_critical_sec:
                if self._can_trigger(EventType.EYE_CLOSED, now):
                    events.append(SuspiciousEvent(
                        event_type=EventType.EYE_CLOSED.value,
                        severity=Severity.CRITICAL.value,
                        message=f"Eyes closed for {duration:.0f}s — "
                                f"possible reading from hidden device",
                        confidence=min(0.9, 0.6 + duration * 0.05),
                    ))
            elif duration >= self.thresholds.eye_closed_warning_sec:
                if self._can_trigger(EventType.EYE_CLOSED, now):
                    events.append(SuspiciousEvent(
                        event_type=EventType.EYE_CLOSED.value,
                        severity=Severity.WARNING.value,
                        message=f"Prolonged eye closure detected ({duration:.0f}s)",
                        confidence=min(0.8, 0.5 + duration * 0.06),
                    ))
        else:
            self._eye_closed_start = None

        return events

    def _check_hand_near_face(self, features: dict, now: float) -> List[SuspiciousEvent]:
        """Detect hand near face (potential phone use or whispering)."""
        events = []
        hand_interaction = features.get("hand_obj_interaction", 0)

        if hand_interaction:
            if self._hand_face_start is None:
                self._hand_face_start = now

            duration = now - self._hand_face_start

            if duration >= self.thresholds.hand_face_warning_sec:
                if self._can_trigger(EventType.HAND_FACE, now):
                    events.append(SuspiciousEvent(
                        event_type=EventType.HAND_FACE.value,
                        severity=Severity.WARNING.value,
                        message=f"Hand near face detected for {duration:.0f}s — "
                                f"possible phone use or communication",
                        confidence=min(0.8, 0.5 + duration * 0.04),
                    ))
        else:
            self._hand_face_start = None

        return events

    # ─── Helpers ──────────────────────────────────────────────────

    def _can_trigger(self, event_type: EventType, now: float) -> bool:
        """Check if the cooldown for this event type has expired."""
        key = event_type.value
        last = self._cooldowns.get(key, 0)
        if now - last >= self.thresholds.cooldown_sec:
            self._cooldowns[key] = now
            return True
        return False

    def get_all_flags(self) -> List[dict]:
        """Return all flagged events as dicts."""
        return [e.to_dict() for e in self.flagged_events]

    def get_warning_summary(self) -> dict:
        """Return a summary of all warnings by type."""
        total = sum(self.warning_counts.values())
        elapsed = time.time() - self.session_start
        return {
            "total_warnings": total,
            "by_type": dict(self.warning_counts),
            "session_duration_sec": round(elapsed, 1),
            "warnings_per_minute": round(total / max(elapsed / 60, 0.01), 2),
        }

    def clear_flags(self):
        """Clear all flagged events (but keep counters)."""
        self.flagged_events.clear()

    def reset(self):
        """Full reset of analyzer state."""
        self._gaze_away_start = None
        self._gaze_directions.clear()
        self._gaze_timestamps.clear()
        self._face_absent_start = None
        self._head_poses.clear()
        self._head_timestamps.clear()
        self._last_head_pose = "forward"
        self._eye_closed_start = None
        self._hand_face_start = None
        self._cooldowns.clear()
        self.flagged_events.clear()
        self.warning_counts = {e.value: 0 for e in EventType}
        self.session_start = time.time()

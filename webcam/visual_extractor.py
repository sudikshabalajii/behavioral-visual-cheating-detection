"""
Visual Feature Extractor (Webcam)
=================================
Uses MediaPipe Face Mesh and Hands to extract visual features from
webcam frames in real-time. Outputs feature dicts that match the
dataset column schema for seamless model inference.

Enhanced with:
  - Eye Aspect Ratio (EAR) for blink / eye closure detection
  - Gaze velocity for rapid eye movement detection
  - Iris-to-eye-corner ratio for precise gaze direction
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time


class VisualExtractor:
    """
    Extracts face landmarks, gaze direction, head pose, and hand presence
    from a single webcam frame using MediaPipe.
    """

    def __init__(self):
        # ── MediaPipe Face Mesh ───────────────────────────────────────
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # includes iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── MediaPipe Hands ───────────────────────────────────────────
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Landmark indices for key facial points (MediaPipe 468+10 iris)
        self.LEFT_EYE_CENTER = 468    # Left iris center
        self.RIGHT_EYE_CENTER = 473   # Right iris center
        self.NOSE_TIP = 1
        self.MOUTH_CENTER = 13
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_OUTER = 263

        # EAR (Eye Aspect Ratio) landmark indices
        # Left eye: 6 landmarks forming the eye contour
        self.LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
        # Right eye: 6 landmarks forming the eye contour
        self.RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

        # Left eye inner/outer corners for iris ratio
        self.LEFT_EYE_INNER = 362
        self.LEFT_EYE_OUTER_IDX = 263
        self.RIGHT_EYE_INNER = 133
        self.RIGHT_EYE_OUTER_IDX = 33

        # 3D model points for head pose estimation (generic face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye corner
            (225.0, 170.0, -135.0),    # Right eye corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0),   # Right mouth corner
        ], dtype=np.float64)

        # Corresponding landmark indices
        self.pose_landmark_ids = [1, 152, 33, 263, 61, 291]

        # Gaze velocity tracking (previous frame gaze point)
        self._prev_gaze_x = None
        self._prev_gaze_y = None
        self._prev_gaze_time = None

    def extract_features(self, frame: np.ndarray) -> dict:
        """
        Extract visual features from a single BGR frame.

        Returns:
            Dictionary with keys matching the dataset schema:
            face_present, no_of_face, face_x, face_y, face_w, face_h,
            left_eye_x/y, right_eye_x/y, nose_tip_x/y, mouth_x/y,
            face_conf, hand_count, left_hand_x/y, right_hand_x/y,
            hand_obj_interaction, head_pose, head_pitch, head_yaw,
            head_roll, phone_present, phone_loc_x/y, phone_conf,
            gaze_on_script, gaze_direction, gazePoint_x/y,
            pupil_left_x/y, pupil_right_x/y
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize default features (no face detected)
        features = self._get_default_features()

        # ── Face Mesh Processing ──────────────────────────────────────
        face_results = self.face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark

            features["face_present"] = 1
            features["no_of_face"] = len(face_results.multi_face_landmarks)

            # Face bounding box (from landmarks)
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            features["face_x"] = x_min
            features["face_y"] = y_min
            features["face_w"] = x_max - x_min
            features["face_h"] = y_max - y_min
            features["face_conf"] = 90.0  # MediaPipe doesn't expose raw conf

            # Eye positions
            left_eye = landmarks[self.LEFT_EYE_CENTER]
            right_eye = landmarks[self.RIGHT_EYE_CENTER]
            features["left_eye_x"] = left_eye.x * w
            features["left_eye_y"] = left_eye.y * h
            features["right_eye_x"] = right_eye.x * w
            features["right_eye_y"] = right_eye.y * h

            # Pupil positions (same as iris centers in MediaPipe)
            features["pupil_left_x"] = int(left_eye.x * w)
            features["pupil_left_y"] = int(left_eye.y * h)
            features["pupil_right_x"] = int(right_eye.x * w)
            features["pupil_right_y"] = int(right_eye.y * h)

            # ── EAR (Eye Aspect Ratio) ────────────────────────────
            try:
                ear_left = self._compute_ear(landmarks, self.LEFT_EYE_EAR, w, h)
                ear_right = self._compute_ear(landmarks, self.RIGHT_EYE_EAR, w, h)
                ear_avg = (ear_left + ear_right) / 2.0
                features["ear_left"] = round(ear_left, 4)
                features["ear_right"] = round(ear_right, 4)
                features["ear_avg"] = round(ear_avg, 4)
                features["eye_open_ratio"] = round(min(ear_avg / 0.30, 1.0), 3)
            except Exception:
                pass  # keep defaults from _get_default_features

            # ── Iris Position Ratio (horizontal) ──────────────────
            try:
                iris_ratio_left = self._compute_iris_ratio(
                    landmarks, self.LEFT_EYE_CENTER,
                    self.LEFT_EYE_INNER, self.LEFT_EYE_OUTER_IDX, w
                )
                iris_ratio_right = self._compute_iris_ratio(
                    landmarks, self.RIGHT_EYE_CENTER,
                    self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER_IDX, w
                )
                features["iris_ratio_left"] = round(iris_ratio_left, 4)
                features["iris_ratio_right"] = round(iris_ratio_right, 4)
                features["iris_ratio_avg"] = round(
                    (iris_ratio_left + iris_ratio_right) / 2.0, 4
                )
            except Exception:
                pass

            # Nose and mouth
            nose = landmarks[self.NOSE_TIP]
            mouth = landmarks[self.MOUTH_CENTER]
            features["nose_tip_x"] = nose.x * w
            features["nose_tip_y"] = nose.y * h
            features["mouth_x"] = mouth.x * w
            features["mouth_y"] = mouth.y * h

            # Head pose estimation
            try:
                pitch, yaw, roll = self._estimate_head_pose(landmarks, w, h)
                features["head_pitch"] = pitch
                features["head_yaw"] = yaw
                features["head_roll"] = roll
                features["head_pose"] = self._classify_head_pose(pitch, yaw)
            except Exception:
                features["head_pose"] = "forward"

            # Gaze direction
            gaze_x, gaze_y = self._estimate_gaze_point(landmarks, w, h)
            features["gazePoint_x"] = int(gaze_x)
            features["gazePoint_y"] = int(gaze_y)
            features["gaze_direction"] = self._classify_gaze(features.get("iris_ratio_avg", 0.5))
            features["gaze_on_script"] = 1 if features["gaze_direction"] == "center" else 0

            # ── Gaze Velocity ─────────────────────────────────────
            try:
                gaze_velocity = self._compute_gaze_velocity(gaze_x, gaze_y)
                features["gaze_velocity"] = round(gaze_velocity, 2)
            except Exception:
                pass

        # ── Hand Detection ────────────────────────────────────────────
        hand_results = self.hands.process(rgb_frame)

        if hand_results.multi_hand_landmarks:
            features["hand_count"] = len(hand_results.multi_hand_landmarks)

            for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                wrist = hand_lm.landmark[0]
                if i == 0:
                    features["left_hand_x"] = wrist.x * w
                    features["left_hand_y"] = wrist.y * h
                elif i == 1:
                    features["right_hand_x"] = wrist.x * w
                    features["right_hand_y"] = wrist.y * h

            # Simple hand-object interaction heuristic:
            # If hand is in the upper face region, likely interacting
            if features["face_present"]:
                for hand_lm in hand_results.multi_hand_landmarks:
                    wrist_y = hand_lm.landmark[0].y * h
                    if wrist_y < features["face_y"] + features["face_h"]:
                        features["hand_obj_interaction"] = 1
                        break

        # Phone detection is not included in MediaPipe
        # (would require a separate object detection model)
        features["phone_present"] = 0
        features["phone_loc_x"] = 0
        features["phone_loc_y"] = 0
        features["phone_conf"] = 0.0

        return features

    def _get_default_features(self) -> dict:
        """Return default feature dict when no face is detected."""
        return {
            "face_present": 0, "no_of_face": 0,
            "face_x": 0.0, "face_y": 0.0, "face_w": 0.0, "face_h": 0.0,
            "left_eye_x": 0.0, "left_eye_y": 0.0,
            "right_eye_x": 0.0, "right_eye_y": 0.0,
            "nose_tip_x": 0.0, "nose_tip_y": 0.0,
            "mouth_x": 0.0, "mouth_y": 0.0,
            "face_conf": 0.0, "hand_count": 0,
            "left_hand_x": 0.0, "left_hand_y": 0.0,
            "right_hand_x": 0.0, "right_hand_y": 0.0,
            "hand_obj_interaction": 0,
            "head_pose": "unknown", "head_pitch": 0.0,
            "head_yaw": 0.0, "head_roll": 0.0,
            "phone_present": 0, "phone_loc_x": 0,
            "phone_loc_y": 0, "phone_conf": 0.0,
            "gaze_on_script": 0, "gaze_direction": "unknown",
            "gazePoint_x": 0, "gazePoint_y": 0,
            "pupil_left_x": 0, "pupil_left_y": 0,
            "pupil_right_x": 0, "pupil_right_y": 0,
            # Enhanced eye metrics
            "ear_left": 0.0, "ear_right": 0.0, "ear_avg": 0.0,
            "eye_open_ratio": 0.0,
            "iris_ratio_left": 0.5, "iris_ratio_right": 0.5,
            "iris_ratio_avg": 0.5,
            "gaze_velocity": 0.0,
        }

    def _estimate_head_pose(self, landmarks, w, h) -> tuple:
        """
        Estimate head pitch, yaw, roll using solvePnP
        with 6 facial landmarks.
        """
        image_points = np.array([
            (landmarks[idx].x * w, landmarks[idx].y * h)
            for idx in self.pose_landmark_ids
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        # Build 3x4 projection matrix, then add [0,0,0,1] row for 4x4
        pose_mat = np.hstack([rotation_mat, translation_vec])
        proj_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1], dtype=np.float64)])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            proj_mat[:3, :]  # decomposeProjectionMatrix expects 3x4
        )

        pitch = euler_angles[0, 0] / 100.0
        yaw = euler_angles[1, 0] / 100.0
        roll = euler_angles[2, 0] / 100.0

        return float(pitch), float(yaw), float(roll)

    def _classify_head_pose(self, pitch: float, yaw: float) -> str:
        """Classify head direction from pitch and yaw angles."""
        if abs(yaw) > 0.15:
            return "right" if yaw > 0 else "left"
        if pitch < -0.15:
            return "down"
        return "forward"

    def _estimate_gaze_point(self, landmarks, w, h) -> tuple:
        """
        Estimate approximate gaze point on screen from iris landmarks.
        """
        left_iris = landmarks[self.LEFT_EYE_CENTER]
        right_iris = landmarks[self.RIGHT_EYE_CENTER]

        gaze_x = ((left_iris.x + right_iris.x) / 2) * w
        gaze_y = ((left_iris.y + right_iris.y) / 2) * h

        return gaze_x, gaze_y

    def _classify_gaze(self, iris_ratio_avg: float) -> str:
        """
        Classify gaze into directional regions using iris ratio.
        """
        if iris_ratio_avg < 0.40:
            return "right"
        elif iris_ratio_avg > 0.60:
            return "left"
        return "center"

    def _compute_ear(self, landmarks, eye_indices, w, h) -> float:
        """
        Compute Eye Aspect Ratio (EAR) for blink detection.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        Where p1-p6 are the 6 eye contour landmarks:
          p1 = outer corner, p4 = inner corner
          p2, p3 = upper lid, p5, p6 = lower lid
        """
        def _dist(a, b):
            ax, ay = landmarks[a].x * w, landmarks[a].y * h
            bx, by = landmarks[b].x * w, landmarks[b].y * h
            return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

        p1, p2, p3, p4, p5, p6 = eye_indices

        # Vertical distances
        vert1 = _dist(p2, p6)
        vert2 = _dist(p3, p5)

        # Horizontal distance
        horiz = _dist(p1, p4)

        if horiz < 1e-6:
            return 0.0

        ear = (vert1 + vert2) / (2.0 * horiz)
        return ear

    def _compute_iris_ratio(self, landmarks, iris_idx, inner_idx, outer_idx, w) -> float:
        """
        Compute horizontal iris position ratio within the eye.

        Returns 0.0 (looking right) to 1.0 (looking left), ~0.5 = center.
        """
        iris_x = landmarks[iris_idx].x * w
        inner_x = landmarks[inner_idx].x * w
        outer_x = landmarks[outer_idx].x * w

        eye_width = abs(outer_x - inner_x)
        if eye_width < 1e-6:
            return 0.5

        ratio = (iris_x - min(inner_x, outer_x)) / eye_width
        return max(0.0, min(1.0, ratio))

    def _compute_gaze_velocity(self, gaze_x: float, gaze_y: float) -> float:
        """
        Compute gaze velocity (pixels/second) from previous frame.
        """
        now = time.time()

        if self._prev_gaze_x is None:
            self._prev_gaze_x = gaze_x
            self._prev_gaze_y = gaze_y
            self._prev_gaze_time = now
            return 0.0

        dt = now - self._prev_gaze_time
        if dt < 1e-6:
            return 0.0

        dx = gaze_x - self._prev_gaze_x
        dy = gaze_y - self._prev_gaze_y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        velocity = distance / dt

        self._prev_gaze_x = gaze_x
        self._prev_gaze_y = gaze_y
        self._prev_gaze_time = now

        return velocity

    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
        self.hands.close()


# ─── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    extractor = VisualExtractor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        exit()

    print("[INFO] Press 'q' to quit webcam test")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = extractor.extract_features(frame)

        # Display overlay
        info_text = [
            f"Face: {features['face_present']}",
            f"Head: {features['head_pose']}",
            f"Gaze: {features['gaze_direction']}",
            f"Hands: {features['hand_count']}",
            f"EAR: {features['ear_avg']:.3f}",
            f"Iris: {features['iris_ratio_avg']:.3f}",
            f"Gaze Vel: {features['gaze_velocity']:.1f}",
        ]
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Visual Extractor Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.release()

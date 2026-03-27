"""
Visual Feature Extractor (Webcam)
=================================
Uses MediaPipe Face Mesh and Hands to extract visual features from
webcam frames in real-time. Outputs feature dicts that match the
dataset column schema for seamless model inference.
"""

import cv2
import numpy as np
import mediapipe as mp
import math


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

            # Nose and mouth
            nose = landmarks[self.NOSE_TIP]
            mouth = landmarks[self.MOUTH_CENTER]
            features["nose_tip_x"] = nose.x * w
            features["nose_tip_y"] = nose.y * h
            features["mouth_x"] = mouth.x * w
            features["mouth_y"] = mouth.y * h

            # Head pose estimation
            pitch, yaw, roll = self._estimate_head_pose(landmarks, w, h)
            features["head_pitch"] = pitch
            features["head_yaw"] = yaw
            features["head_roll"] = roll
            features["head_pose"] = self._classify_head_pose(pitch, yaw)

            # Gaze direction
            gaze_x, gaze_y = self._estimate_gaze_point(landmarks, w, h)
            features["gazePoint_x"] = int(gaze_x)
            features["gazePoint_y"] = int(gaze_y)
            features["gaze_direction"] = self._classify_gaze(gaze_x, gaze_y, w, h)
            features["gaze_on_script"] = 1 if features["gaze_direction"] == "center" else 0

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
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            cv2.hconcat([pose_mat, np.array([[0, 0, 0, 1]])])
        )

        pitch = euler_angles[0, 0] / 100.0
        yaw = euler_angles[1, 0] / 100.0
        roll = euler_angles[2, 0] / 100.0

        return float(pitch), float(yaw), float(roll)

    def _classify_head_pose(self, pitch: float, yaw: float) -> str:
        """Classify head direction from pitch and yaw angles."""
        if abs(yaw) > 0.03:
            return "right" if yaw > 0 else "left"
        if pitch < -0.02:
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

    def _classify_gaze(self, gx, gy, w, h) -> str:
        """
        Classify gaze into directional regions.
        Screen divided into 3×2 grid: top/bottom × left/center/right.
        """
        x_third = w / 3
        y_half = h / 2

        if gx < x_third:
            h_dir = "left"
        elif gx > 2 * x_third:
            h_dir = "right"
        else:
            h_dir = "center"

        v_dir = "top" if gy < y_half else "bottom"

        if h_dir == "center" and abs(gy - y_half) < y_half * 0.3:
            return "center"

        return f"{v_dir}_{h_dir}" if h_dir != "center" else "center"

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
            f"Pitch: {features['head_pitch']:.3f}",
            f"Yaw: {features['head_yaw']:.3f}",
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

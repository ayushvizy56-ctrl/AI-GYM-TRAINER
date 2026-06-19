import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False


class PoseDetector:
    KEYPOINTS = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13,    "right_elbow": 14,
        "left_wrist": 15,    "right_wrist": 16,
        "left_hip": 23,      "right_hip": 24,
        "left_knee": 25,     "right_knee": 26,
        "left_ankle": 27,    "right_ankle": 28,
    }

    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        if not MEDIAPIPE_AVAILABLE:
            self.pose = None
            print("⚠ MediaPipe not available")
            return

        try:
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            print("✓ MediaPipe loaded successfully")
        except Exception as e:
            print(f"⚠ MediaPipe init failed: {e}")
            self.pose = None

    def process(self, frame: np.ndarray):
        if self.pose is None:
            return None, frame

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.pose.process(rgb)
            rgb.flags.writeable = True
            annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if not results.pose_landmarks:
                return None, annotated

            self.mp_draw.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )

            h, w, _ = frame.shape
            raw = results.pose_landmarks.landmark
            landmarks = {}
            for name, idx in self.KEYPOINTS.items():
                lm = raw[idx]
                landmarks[name] = {
                    "x": lm.x * w,
                    "y": lm.y * h,
                    "z": lm.z,
                    "visibility": lm.visibility,
                }

            return landmarks, annotated

        except Exception as e:
            print(f"Pose detection error: {e}")
            return None, frame

    def close(self):
        if self.pose:
            self.pose.close()
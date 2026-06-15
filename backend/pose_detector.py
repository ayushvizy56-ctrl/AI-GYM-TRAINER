"""
pose_detector.py
-----------------
Uses MediaPipe BlazePose to detect 33 body landmarks from a camera frame.
Returns landmark coordinates and draws a skeleton overlay on the frame.
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """
    Wraps MediaPipe Pose for real-time body landmark detection.

    Usage:
        detector = PoseDetector()
        landmarks, annotated_frame = detector.process(frame)
    """

    # MediaPipe landmark indices we care about most
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
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            model_complexity=1,                       # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray):
        """
        Process a BGR frame and return landmarks + annotated frame.

        Returns:
            landmarks (dict | None): {name: {x, y, z, visibility}}
            annotated_frame (np.ndarray): Frame with skeleton drawn
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            return None, annotated

        # Draw skeleton on frame
        self.mp_draw.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
        )

        # Extract named landmarks
        h, w, _ = frame.shape
        raw = results.pose_landmarks.landmark
        landmarks = {}
        for name, idx in self.KEYPOINTS.items():
            lm = raw[idx]
            landmarks[name] = {
                "x": lm.x * w,   # pixel coords
                "y": lm.y * h,
                "z": lm.z,
                "visibility": lm.visibility,
            }

        return landmarks, annotated

    def close(self):
        self.pose.close()

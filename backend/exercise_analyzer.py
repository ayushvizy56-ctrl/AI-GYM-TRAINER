"""
exercise_analyzer.py
---------------------
Calculates joint angles from pose landmarks and grades exercise form.

For each supported exercise we define:
  - Which joint angle(s) to compute
  - "Good form" angle ranges
  - Common mistakes and their corrections
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FormResult:
    exercise: str
    rep_phase: str          # "up", "down", "hold"
    angles: dict            # joint_name -> degrees
    score: int              # 0-100
    status: str             # "good", "warning", "error"
    feedback: list          # list of feedback strings
    corrections: list       # list of corrections
    key_angle: float = 0.0  # the primary angle driving rep count


# ── Angle helpers ────────────────────────────────────────────────────────────

def _angle(a, b, c) -> float:
    """
    Calculate the angle at joint B formed by points A-B-C.
    Returns degrees (0-180).
    """
    a = np.array([a["x"], a["y"]])
    b = np.array([b["x"], b["y"]])
    c = np.array([c["x"], c["y"]])

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _midpoint(p1, p2):
    return {
        "x": (p1["x"] + p2["x"]) / 2,
        "y": (p1["y"] + p2["y"]) / 2,
        "z": (p1["z"] + p2["z"]) / 2,
        "visibility": min(p1["visibility"], p2["visibility"]),
    }


# ── Exercise analyzers ───────────────────────────────────────────────────────

class ExerciseAnalyzer:
    """
    Analyzes exercise form given landmarks from PoseDetector.

    Supported exercises:
        squat, bicep_curl, pushup, shoulder_press, deadlift, lunge
    """

    EXERCISES = ["squat", "bicep_curl", "pushup", "shoulder_press", "deadlift", "lunge"]

    def analyze(self, exercise: str, landmarks: dict) -> Optional[FormResult]:
        if landmarks is None:
            return None

        fn = getattr(self, f"_analyze_{exercise}", None)
        if fn is None:
            return None
        return fn(landmarks)

    # ── Squat ────────────────────────────────────────────────────────────────
    def _analyze_squat(self, lm) -> FormResult:
        angles = {}
        feedback = []
        corrections = []
        score = 100

        # Left + right knee angles
        lk = _angle(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
        rk = _angle(lm["right_hip"], lm["right_knee"], lm["right_ankle"])
        angles["left_knee"] = round(lk, 1)
        angles["right_knee"] = round(rk, 1)

        # Hip angle (back angle)
        lh = _angle(lm["left_shoulder"], lm["left_hip"], lm["left_knee"])
        rh = _angle(lm["right_shoulder"], lm["right_hip"], lm["right_knee"])
        angles["left_hip"] = round(lh, 1)
        angles["right_hip"] = round(rh, 1)

        avg_knee = (lk + rk) / 2
        avg_hip = (lh + rh) / 2

        # Determine phase
        if avg_knee < 110:
            phase = "down"
        elif avg_knee > 155:
            phase = "up"
        else:
            phase = "mid"

        # --- Form checks ---
        # Depth
        if phase == "down" and avg_knee > 100:
            feedback.append("Go deeper — aim for thighs parallel to floor")
            corrections.append("depth")
            score -= 15

        # Knee caving (valgus)
        knee_dist = abs(lm["left_knee"]["x"] - lm["right_knee"]["x"])
        ankle_dist = abs(lm["left_ankle"]["x"] - lm["right_ankle"]["x"])
        if ankle_dist > 0 and knee_dist / ankle_dist < 0.75:
            feedback.append("Knees caving in — push them out over toes")
            corrections.append("knee_valgus")
            score -= 20

        # Forward lean
        if avg_hip < 50:
            feedback.append("Excessive forward lean — keep chest up")
            corrections.append("forward_lean")
            score -= 15

        # Symmetry
        if abs(lk - rk) > 15:
            feedback.append("Asymmetric squat — balance weight evenly")
            corrections.append("asymmetry")
            score -= 10

        if score >= 85:
            status = "good"
            if not feedback:
                feedback.append("Great squat form!")
        elif score >= 60:
            status = "warning"
        else:
            status = "error"

        return FormResult(
            exercise="squat", rep_phase=phase,
            angles=angles, score=max(0, score),
            status=status, feedback=feedback, corrections=corrections,
            key_angle=avg_knee
        )

    # ── Bicep Curl ───────────────────────────────────────────────────────────
    def _analyze_bicep_curl(self, lm) -> FormResult:
        angles = {}
        feedback = []
        corrections = []
        score = 100

        le = _angle(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])
        re = _angle(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"])
        angles["left_elbow"] = round(le, 1)
        angles["right_elbow"] = round(re, 1)

        avg_elbow = (le + re) / 2
        phase = "up" if avg_elbow < 60 else "down" if avg_elbow > 150 else "mid"

        # Elbow swinging (shoulder angle should stay stable)
        ls = _angle(lm["left_hip"], lm["left_shoulder"], lm["left_elbow"])
        rs = _angle(lm["right_hip"], lm["right_shoulder"], lm["right_elbow"])
        angles["left_shoulder"] = round(ls, 1)
        angles["right_shoulder"] = round(rs, 1)

        if ls < 140 or rs < 140:
            feedback.append("Keep elbows pinned to sides — avoid swinging")
            corrections.append("elbow_swinging")
            score -= 25

        # Full range of motion
        if phase == "up" and avg_elbow > 50:
            feedback.append("Curl higher for full contraction")
            corrections.append("partial_rom")
            score -= 10

        if score >= 85:
            status = "good"
            if not feedback:
                feedback.append("Clean curl — full range, no swing!")
        elif score >= 60:
            status = "warning"
        else:
            status = "error"

        return FormResult(
            exercise="bicep_curl", rep_phase=phase,
            angles=angles, score=max(0, score),
            status=status, feedback=feedback, corrections=corrections,
            key_angle=avg_elbow
        )

    # ── Push-up ──────────────────────────────────────────────────────────────
    def _analyze_pushup(self, lm) -> FormResult:
        angles = {}
        feedback = []
        corrections = []
        score = 100

        le = _angle(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])
        re = _angle(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"])
        angles["left_elbow"] = round(le, 1)
        angles["right_elbow"] = round(re, 1)
        avg_elbow = (le + re) / 2

        # Body alignment (hip shouldn't sag or pike)
        hip_mid = _midpoint(lm["left_hip"], lm["right_hip"])
        shoulder_mid = _midpoint(lm["left_shoulder"], lm["right_shoulder"])
        ankle_mid = _midpoint(lm["left_ankle"], lm["right_ankle"])
        body_angle = _angle(shoulder_mid, hip_mid, ankle_mid)
        angles["body_line"] = round(body_angle, 1)

        phase = "down" if avg_elbow < 90 else "up"

        if body_angle < 160:
            if hip_mid["y"] < shoulder_mid["y"]:
                feedback.append("Hips too high — lower into a straight plank")
                corrections.append("hip_pike")
            else:
                feedback.append("Hips sagging — engage your core")
                corrections.append("hip_sag")
            score -= 25

        if phase == "down" and avg_elbow > 95:
            feedback.append("Go lower — chest should nearly touch floor")
            corrections.append("depth")
            score -= 15

        if score >= 85:
            status = "good"
            if not feedback:
                feedback.append("Perfect push-up form!")
        elif score >= 60:
            status = "warning"
        else:
            status = "error"

        return FormResult(
            exercise="pushup", rep_phase=phase,
            angles=angles, score=max(0, score),
            status=status, feedback=feedback, corrections=corrections,
            key_angle=avg_elbow
        )

    # ── Shoulder Press ───────────────────────────────────────────────────────
    def _analyze_shoulder_press(self, lm) -> FormResult:
        angles = {}
        feedback = []
        corrections = []
        score = 100

        le = _angle(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])
        re = _angle(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"])
        angles["left_elbow"] = round(le, 1)
        angles["right_elbow"] = round(re, 1)
        avg_elbow = (le + re) / 2

        ls = _angle(lm["left_hip"], lm["left_shoulder"], lm["left_elbow"])
        rs = _angle(lm["right_hip"], lm["right_shoulder"], lm["right_elbow"])
        angles["left_shoulder_abduction"] = round(ls, 1)
        angles["right_shoulder_abduction"] = round(rs, 1)

        phase = "up" if avg_elbow > 160 else "down"

        if ls < 70 or rs < 70:
            feedback.append("Elbows dropping — raise them to shoulder height")
            corrections.append("elbow_drop")
            score -= 20

        if score >= 85:
            status = "good"
            if not feedback:
                feedback.append("Solid press — arms fully extended!")
        elif score >= 60:
            status = "warning"
        else:
            status = "error"

        return FormResult(
            exercise="shoulder_press", rep_phase=phase,
            angles=angles, score=max(0, score),
            status=status, feedback=feedback, corrections=corrections,
            key_angle=avg_elbow
        )

    # ── Deadlift ─────────────────────────────────────────────────────────────
    def _analyze_deadlift(self, lm) -> FormResult:
        angles = {}
        feedback = []
        corrections = []
        score = 100

        back_angle = _angle(lm["left_shoulder"], lm["left_hip"], lm["left_knee"])
        angles["back_angle"] = round(back_angle, 1)

        lk = _angle(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
        angles["left_knee"] = round(lk, 1)

        phase = "up" if back_angle > 160 else "down"

        if back_angle < 140 and phase == "down":
            feedback.append("Keep back neutral — don't round the lower back")
            corrections.append("back_rounding")
            score -= 30

        if lk < 140 and phase == "up":
            feedback.append("Fully extend knees at the top")
            corrections.append("knee_lockout")
            score -= 10

        if score >= 85:
            status = "good"
            if not feedback:
                feedback.append("Strong pull — neutral spine throughout!")
        elif score >= 60:
            status = "warning"
        else:
            status = "error"

        return FormResult(
            exercise="deadlift", rep_phase=phase,
            angles=angles, score=max(0, score),
            status=status, feedback=feedback, corrections=corrections,
            key_angle=back_angle
        )

    # ── Lunge ────────────────────────────────────────────────────────────────
    def _analyze_lunge(self, lm) -> FormResult:
        angles = {}
        feedback = []
        corrections = []
        score = 100

        lk = _angle(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
        rk = _angle(lm["right_hip"], lm["right_knee"], lm["right_ankle"])
        angles["front_knee"] = round(min(lk, rk), 1)
        angles["back_knee"] = round(max(lk, rk), 1)

        front_knee = min(lk, rk)
        phase = "down" if front_knee < 100 else "up"

        if phase == "down" and front_knee > 95:
            feedback.append("Bend front knee to 90° for full depth")
            corrections.append("depth")
            score -= 15

        torso = _angle(lm["left_shoulder"], lm["left_hip"], lm["left_knee"])
        if torso < 160:
            feedback.append("Keep torso upright — don't lean forward")
            corrections.append("forward_lean")
            score -= 20

        if score >= 85:
            status = "good"
            if not feedback:
                feedback.append("Great lunge — balanced and controlled!")
        elif score >= 60:
            status = "warning"
        else:
            status = "error"

        return FormResult(
            exercise="lunge", rep_phase=phase,
            angles=angles, score=max(0, score),
            status=status, feedback=feedback, corrections=corrections,
            key_angle=front_knee
        )

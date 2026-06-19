import base64
import json
import os
import time
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from pose_detector import PoseDetector
from exercise_analyzer import ExerciseAnalyzer
from rep_counter import RepCounter
from ai_coach import AICoach

app = Flask(__name__)
CORS(app)

detector = PoseDetector()
analyzer = ExerciseAnalyzer()
counter = RepCounter()

try:
    coach = AICoach()
    COACH_AVAILABLE = True
    print("✓ Claude AI coach connected")
except Exception:
    coach = None
    COACH_AVAILABLE = False
    print("⚠  AI coaching disabled - set ANTHROPIC_API_KEY")

sessions: dict = {}


def decode_frame(b64_image: str) -> np.ndarray:
    if "," in b64_image:
        b64_image = b64_image.split(",")[1]
    img_bytes = base64.b64decode(b64_image)
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (320, 240))
    return frame


def encode_frame(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


@app.route("/api/exercises", methods=["GET"])
def get_exercises():
    return jsonify({
        "exercises": [
            {"id": "squat",          "name": "Squat",          "muscle": "Legs",      "icon": "🦵"},
            {"id": "bicep_curl",     "name": "Bicep Curl",     "muscle": "Arms",      "icon": "💪"},
            {"id": "pushup",         "name": "Push-up",        "muscle": "Chest",     "icon": "🤸"},
            {"id": "shoulder_press", "name": "Shoulder Press", "muscle": "Shoulders", "icon": "🏋️"},
            {"id": "deadlift",       "name": "Deadlift",       "muscle": "Back",      "icon": "⬆️"},
            {"id": "lunge",          "name": "Lunge",          "muscle": "Legs",      "icon": "🚶"},
        ]
    })


@app.route("/api/analyze-frame", methods=["POST"])
def analyze_frame():
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "Missing frame"}), 400

    exercise = data.get("exercise", "squat")
    session_id = data.get("session_id", "default")
    t_start = time.time()

    try:
        frame = decode_frame(data["frame"])
    except Exception as e:
        return jsonify({"error": f"Frame decode failed: {e}"}), 400

    landmarks, annotated = detector.process(frame)

    if landmarks is None:
        return jsonify({
            "annotated_frame": encode_frame(annotated),
            "landmarks_detected": False,
            "form": None,
            "reps": counter.update(exercise, 0),
            "processing_ms": int((time.time() - t_start) * 1000),
        })

    form_result = analyzer.analyze(exercise, landmarks)
    rep_data = counter.update(exercise, form_result.key_angle)

    if session_id not in sessions:
        sessions[session_id] = {"scores": [], "exercise": exercise, "start_time": time.time()}
    sessions[session_id]["scores"].append(form_result.score)
    sessions[session_id]["last_rep"] = rep_data["count"]

    _draw_hud(annotated, form_result, rep_data)

    processing_ms = int((time.time() - t_start) * 1000)

    return jsonify({
        "annotated_frame": encode_frame(annotated),
        "landmarks_detected": True,
        "form": {
            "score": form_result.score,
            "status": form_result.status,
            "feedback": form_result.feedback,
            "corrections": form_result.corrections,
            "angles": form_result.angles,
            "rep_phase": form_result.rep_phase,
        },
        "reps": rep_data,
        "processing_ms": processing_ms,
    })


@app.route("/api/coaching", methods=["POST"])
def get_coaching():
    if not COACH_AVAILABLE:
        return jsonify({"error": "AI coaching not available - set ANTHROPIC_API_KEY"}), 503

    data = request.get_json()
    session_id = data.get("session_id", "default")
    history = sessions.get(session_id, {}).get("scores", [])

    try:
        advice = coach.get_coaching(
            exercise=data.get("exercise", "squat"),
            form_score=data.get("form_score", 80),
            corrections=data.get("corrections", []),
            angles=data.get("angles", {}),
            rep_count=data.get("rep_count", 0),
            set_number=data.get("set_number", 1),
            user_level=data.get("user_level", "intermediate"),
            session_reps_history=history,
        )
        return jsonify(advice)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/workout-plan", methods=["POST"])
def workout_plan():
    if not COACH_AVAILABLE:
        return jsonify({"error": "Set ANTHROPIC_API_KEY"}), 503

    data = request.get_json()
    try:
        plan = coach.generate_workout_plan(
            user_profile=data.get("user_profile", {}),
            recent_sessions=data.get("recent_sessions", []),
        )
        return jsonify(plan)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session-report", methods=["POST"])
def session_report():
    if not COACH_AVAILABLE:
        return jsonify({"error": "Set ANTHROPIC_API_KEY"}), 503

    data = request.get_json()
    session_id = data.get("session_id", "default")
    session_data = sessions.get(session_id, {})
    session_data.update(data)

    try:
        report = coach.analyze_session(session_data)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json() or {}
    exercise = data.get("exercise")
    counter.reset(exercise)
    return jsonify({"status": "reset", "exercise": exercise or "all"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "coach_available": COACH_AVAILABLE,
        "model": "claude-sonnet-4-6"
    })


def _draw_hud(frame, form_result, rep_data):
    h, w = frame.shape[:2]

    color = {
        "good": (0, 200, 0),
        "warning": (0, 165, 255),
        "error": (0, 0, 255)
    }.get(form_result.status, (255, 255, 255))

    cv2.putText(frame, f"Reps: {rep_data['count']}", (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2)

    score_text = f"Form: {form_result.score}/100"
    (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.putText(frame, score_text, (w - tw - 20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    cv2.putText(frame, f"Phase: {rep_data['phase']}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    if form_result.feedback:
        msg = form_result.feedback[0]
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, msg, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


if __name__ == "__main__":
    print("🏋️  AI Gym Trainer backend starting...")
    print(f"   Coach: {'✓ Claude AI' if COACH_AVAILABLE else '✗ Fallback mode'}")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False) 
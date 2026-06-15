# 🏋️ AI Gym Trainer

A full-stack AI-powered personal trainer that watches your exercise form in real time,
counts reps, grades your form using computer vision, and gives personalised coaching
advice powered by the Claude AI API.

---

## What it does

| Feature | Technology |
|---|---|
| Body pose detection | MediaPipe BlazePose (33 keypoints, 30fps) |
| Joint angle calculation | NumPy geometry |
| Rep counting | EMA-smoothed state machine + LSTM scaffold |
| Form grading | Rule-based expert system per exercise |
| AI coaching | Claude Sonnet API |
| Voice feedback | Web Speech API (text-to-speech) |
| Charts | Chart.js |

---

## Project structure

```
ai-gym-trainer/
├── backend/
│   ├── app.py                  ← Flask API server (start this)
│   ├── pose_detector.py        ← MediaPipe body keypoint detection
│   ├── exercise_analyzer.py    ← Joint angles + form grading
│   ├── rep_counter.py          ← Rep counting + LSTM scaffold
│   ├── ai_coach.py             ← Claude API coaching
│   └── requirements.txt
├── frontend/
│   ├── index.html              ← Main UI
│   └── trainer.js              ← Camera, API calls, charts
└── README.md
```

---

## Quick start

### Step 1 — Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

> Requires Python 3.10+ and a camera/webcam.

### Step 2 — Set your Anthropic API key

```bash
# Mac/Linux
export ANTHROPIC_API_KEY=your_key_here

# Windows
set ANTHROPIC_API_KEY=your_key_here
```

Get a free API key at: https://console.anthropic.com

### Step 3 — Start the backend

```bash
cd backend
python app.py
```

You should see:
```
🏋️  AI Gym Trainer backend starting...
   Coach: ✓ Claude AI
 * Running on http://0.0.0.0:5000
```

### Step 4 — Open the frontend

Open `frontend/index.html` in your browser.
Or serve it with a simple HTTP server:

```bash
cd frontend
python -m http.server 8080
# Then open: http://localhost:8080
```

> **Important**: The browser must be served over HTTP (not opened as a file) for camera access to work.

---

## How to use

1. **Select an exercise** from the left sidebar
2. Click **▶ Start Camera**
3. Stand 2–3 metres from the camera so your full body is visible
4. **Perform the exercise** — reps are counted automatically
5. Watch the **form score** and **AI coaching** in real time
6. Click **🤖 Ask AI Coach** for personalised feedback
7. Click **📅 Workout Plan** to generate a weekly plan

---

## Supported exercises

| Exercise | Muscles | What's measured |
|---|---|---|
| Squat | Legs, glutes | Knee angle, depth, knee alignment |
| Bicep curl | Biceps | Elbow angle, elbow stability |
| Push-up | Chest, triceps | Elbow angle, body alignment |
| Shoulder press | Shoulders | Elbow extension, shoulder abduction |
| Deadlift | Back, hamstrings | Back angle, neutral spine |
| Lunge | Legs | Knee angle, torso upright |

---

## Adding a new exercise

1. Add form analysis in `backend/exercise_analyzer.py`:
   ```python
   def _analyze_my_exercise(self, lm) -> FormResult:
       # Calculate angles
       # Check form rules
       # Return FormResult(...)
   ```

2. Add rep counting thresholds in `rep_counter.py`:
   ```python
   THRESHOLDS = {
       ...
       "my_exercise": (80, 160),  # (low_threshold, high_threshold)
   }
   ```

3. Add to the exercise list in `app.py` and `trainer.js`

---

## Training the LSTM rep counter (advanced)

The `rep_counter.py` file includes a `train_lstm_counter()` function.
To use it:

1. Record a video of yourself doing 50+ reps of an exercise
2. Run `pose_detector.py` on the video to extract angle sequences
3. Label each frame as "up" or "down" phase
4. Run `train_lstm_counter(X_sequences, y_phases, exercise_name)`
5. The trained `.h5` model will be saved to `models/`

---

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/exercises` | GET | List supported exercises |
| `/api/analyze-frame` | POST | Process camera frame |
| `/api/coaching` | POST | Get AI coaching |
| `/api/workout-plan` | POST | Generate weekly plan |
| `/api/session-report` | POST | Post-session analysis |
| `/api/reset` | POST | Reset rep counter |
| `/health` | GET | Backend health check |

---

## Tech stack

- **Backend**: Python, Flask, MediaPipe, OpenCV, NumPy, TensorFlow, Anthropic SDK
- **Frontend**: Vanilla JS, Chart.js, Web Speech API
- **AI**: Claude Sonnet (claude-sonnet-4-6)

---

## Troubleshooting

**Camera not working**: Make sure you're serving the frontend over HTTP, not opening the file directly.

**Backend not connecting**: Check CORS — the Flask app has `flask-cors` enabled for all origins.

**Pose not detecting**: Make sure your full body is visible in the frame, with good lighting.

**AI coaching not available**: Set the `ANTHROPIC_API_KEY` environment variable.

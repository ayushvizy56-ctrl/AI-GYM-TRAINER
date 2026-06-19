/**
 * trainer.js
 * -----------
 * Handles:
 *  - Camera access and frame capture
 *  - Sending frames to backend for pose analysis
 *  - Updating the UI with form scores, reps, angles
 *  - Text-to-speech coaching
 *  - Chart.js angle history graph
 */

const API = 'https://ai-gym-trainer-z20n.onrender.com/api';
const SESSION_ID = 'session_' + Date.now();

// ── State ─────────────────────────────────────────────────────────────────────
let state = {
  cameraActive: false,
  paused: false,
  currentExercise: 'squat',
  repCount: 0,
  totalReps: 0,
  setCount: 0,
  formScores: [],
  angleHistory: [],
  lastCoachingRep: -1,
  processingFrame: false,
  frameInterval: null,
  stream: null,
  lastSpeech: '',
  speechCooldown: 0,
};

// ── Exercise definitions ─────────────────────────────────────────────────────
const exercises = [
  { id: 'squat',          name: 'Squat',           muscle: 'Legs',      icon: '🦵' },
  { id: 'bicep_curl',     name: 'Bicep Curl',      muscle: 'Arms',      icon: '💪' },
  { id: 'pushup',         name: 'Push-up',         muscle: 'Chest',     icon: '🤸' },
  { id: 'shoulder_press', name: 'Shoulder Press',  muscle: 'Shoulders', icon: '🏋️' },
  { id: 'deadlift',       name: 'Deadlift',        muscle: 'Back',      icon: '⬆️' },
  { id: 'lunge',          name: 'Lunge',           muscle: 'Legs',      icon: '🚶' },
];

// Phase descriptions
const phaseDesc = {
  top:    'Top position — ready to start the movement',
  bottom: 'Bottom position — maximum contraction',
  mid:    'Mid-range — keep controlled movement',
  up:     'Moving up — squeeze the muscle',
  down:   'Going down — control the negative',
  idle:   'No movement detected',
};

// ── Chart setup ───────────────────────────────────────────────────────────────
let angleChart;

function initChart() {
  const ctx = document.getElementById('angle-chart').getContext('2d');
  angleChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: '#6c63ff',
        backgroundColor: 'rgba(108,99,255,0.08)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: {
          display: true,
          min: 0, max: 180,
          ticks: { color: '#8888aa', font: { size: 10 }, maxTicksLimit: 4 },
          grid: { color: 'rgba(255,255,255,0.04)' },
        }
      },
      animation: { duration: 0 }
    }
  });
}

// ── Build exercise sidebar ────────────────────────────────────────────────────
function buildExerciseList() {
  const list = document.getElementById('exercise-list');
  list.innerHTML = exercises.map(ex => `
    <div class="exercise-card ${ex.id === state.currentExercise ? 'active' : ''}"
         onclick="selectExercise('${ex.id}')" id="ex-${ex.id}">
      <div class="exercise-icon">${ex.icon}</div>
      <div class="exercise-info">
        <div class="exercise-name">${ex.name}</div>
        <div class="exercise-muscle">${ex.muscle}</div>
      </div>
    </div>
  `).join('');
}

function selectExercise(id) {
  state.currentExercise = id;
  document.querySelectorAll('.exercise-card').forEach(c => c.classList.remove('active'));
  document.getElementById('ex-' + id)?.classList.add('active');
  document.getElementById('exercise-title').textContent = exercises.find(e => e.id === id)?.name || '';
  resetReps();
}

// ── Set tracker ───────────────────────────────────────────────────────────────
function buildSetTracker() {
  const target = 4;
  const tracker = document.getElementById('set-tracker');
  tracker.innerHTML = Array.from({length: target}, (_, i) => `
    <div class="set-item ${i < state.setCount ? 'done' : ''}" id="set-${i}">
      Set ${i + 1}
    </div>
  `).join('');
}

function completeSet() {
  state.setCount++;
  state.totalReps += state.repCount;
  state.repCount = 0;
  document.getElementById('stat-sets').textContent = state.setCount;
  updateStatsTotalReps();
  buildSetTracker();
  resetRepsOnly();
  speak(`Set ${state.setCount} complete! Great work!`);
}

// ── Camera ────────────────────────────────────────────────────────────────────
async function toggleCamera() {
  if (state.cameraActive) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });

    const video = document.getElementById('video');
    video.srcObject = state.stream;
    await video.play();

    state.cameraActive = true;
    document.getElementById('no-camera').style.display = 'none';
    document.getElementById('hud').style.display = 'flex';
    document.getElementById('btn-camera').textContent = '⏹ Stop Camera';
    document.getElementById('btn-pause').style.display = '';
    setStatus('active', 'Live');

    // Start frame processing loop
    state.frameInterval = setInterval(processFrame, 100); // 10fps
  } catch (err) {
    alert('Camera access denied: ' + err.message);
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  clearInterval(state.frameInterval);
  state.cameraActive = false;
  document.getElementById('no-camera').style.display = 'flex';
  document.getElementById('hud').style.display = 'none';
  document.getElementById('btn-camera').textContent = '▶ Start Camera';
  document.getElementById('btn-pause').style.display = 'none';
  setStatus('idle', 'Stopped');
}

function togglePause() {
  state.paused = !state.paused;
  document.getElementById('btn-pause').textContent = state.paused ? '▶ Resume' : '⏸ Pause';
  setStatus(state.paused ? 'warn' : 'active', state.paused ? 'Paused' : 'Live');
}

// ── Frame processing ──────────────────────────────────────────────────────────
let frameCount = 0;
let fpsTimer = Date.now();

async function processFrame() {
  if (!state.cameraActive || state.paused || state.processingFrame) return;

  const video = document.getElementById('video');
  if (video.readyState < 2) return;

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Capture frame
  ctx.drawImage(video, 0, 0);
  const b64 = canvas.toDataURL('image/jpeg', 0.7);

  state.processingFrame = true;

  try {
    const res = await fetch(`${API}/analyze-frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame: b64,
        exercise: state.currentExercise,
        session_id: SESSION_ID,
      }),
    });

    if (!res.ok) return;
    const data = await res.json();

    // Draw annotated frame back to canvas
    if (data.annotated_frame) {
      const img = new Image();
      img.onload = () => { ctx.drawImage(img, 0, 0); };
      img.src = data.annotated_frame;
    }

    if (data.landmarks_detected && data.form) {
      updateUI(data);
    } else {
      document.getElementById('hud-feedback').textContent = 'No pose detected — step back from camera';
    }

    // FPS counter
    frameCount++;
    if (Date.now() - fpsTimer > 2000) {
      document.getElementById('fps-display').textContent = `${Math.round(frameCount / 2)} fps`;
      frameCount = 0;
      fpsTimer = Date.now();
    }
  } catch (e) {
    // Backend not running — show demo mode
    document.getElementById('coach-cue').textContent = 'Backend not running. Start app.py to connect.';
  } finally {
    state.processingFrame = false;
  }
}

// ── UI updates ────────────────────────────────────────────────────────────────
function updateUI(data) {
  const form = data.form;
  const reps = data.reps;

  // Rep count
  const newCount = reps.count;
  if (newCount !== state.repCount) {
    state.repCount = newCount;
    document.getElementById('hud-reps').textContent = newCount;
    updateStatsTotalReps();

    // Auto AI coaching every N reps
    const targetReps = parseInt(document.getElementById('target-reps').value);
    const shouldCoach = document.getElementById('auto-coach-toggle').checked &&
                        newCount > 0 && newCount % 3 === 0 &&
                        newCount !== state.lastCoachingRep;
    if (shouldCoach) {
      state.lastCoachingRep = newCount;
      triggerAutoCoaching(form, reps);
    }

    // Speak rep milestone
    if (newCount === targetReps) {
      speak(`${targetReps} reps done! Great set!`);
    } else if (newCount > 0 && newCount % 5 === 0) {
      speak(`${newCount} reps!`);
    }
  }

  // Form score
  const score = form.score;
  state.formScores.push(score);
  const avgScore = Math.round(state.formScores.reduce((a,b)=>a+b,0) / state.formScores.length);

  document.getElementById('hud-score').textContent = score;
  document.getElementById('hud-score').className = 'hud-score ' + scoreClass(score);
  document.getElementById('form-score-big').textContent = score;
  document.getElementById('form-score-big').style.color = scoreColor(score);
  document.getElementById('stat-avg-score').textContent = avgScore;

  const fill = document.getElementById('form-bar-fill');
  fill.style.width = score + '%';
  fill.style.background = scoreColor(score);

  document.getElementById('form-status-label').textContent =
    score >= 85 ? '✓ Excellent form' :
    score >= 60 ? '⚠ Needs improvement' : '✗ Fix form before continuing';

  // Feedback text
  const feedback = form.feedback?.[0] || '';
  document.getElementById('hud-feedback').textContent = feedback;

  // Speak feedback (not too often)
  if (feedback && Date.now() > state.speechCooldown) {
    if (form.status === 'error' || (form.status === 'warning' && feedback !== state.lastSpeech)) {
      speak(feedback);
      state.lastSpeech = feedback;
      state.speechCooldown = Date.now() + 4000;
    }
  }

  // Phase indicators
  const phase = reps.phase;
  ['bottom','mid','top'].forEach(p => {
    document.getElementById(`pd-${p}`)?.classList.toggle('active', p === phase);
  });
  document.getElementById('phase-indicator').textContent =
    { top: 'Top ↑', bottom: 'Bottom ↓', mid: 'Mid ↔', up: 'Going Up ↑', down: 'Going Down ↓', idle: 'Idle' }[phase] || phase;
  document.getElementById('phase-desc').textContent = phaseDesc[phase] || '';

  // Joint angles
  updateAngles(form.angles);

  // Angle history chart
  state.angleHistory.push(reps.smoothed_angle);
  if (state.angleHistory.length > 60) state.angleHistory.shift();
  angleChart.data.labels = state.angleHistory.map((_, i) => i);
  angleChart.data.datasets[0].data = [...state.angleHistory];
  angleChart.update('none');

  // Correction pills
  const pillContainer = document.getElementById('correction-pills');
  if (form.corrections?.length) {
    pillContainer.innerHTML = form.corrections.map(c =>
      `<span class="correction-pill ${form.status === 'error' ? 'error' : ''}">${c.replace(/_/g,' ')}</span>`
    ).join('');
  } else {
    pillContainer.innerHTML = '<span style="font-size:12px;color:var(--green);">✓ No corrections needed</span>';
  }
}

function updateAngles(angles) {
  if (!angles || Object.keys(angles).length === 0) return;
  const grid = document.getElementById('angles-grid');
  grid.innerHTML = Object.entries(angles).map(([joint, deg]) => `
    <div class="angle-row">
      <span class="angle-name">${joint.replace(/_/g,' ')}</span>
      <span class="angle-value" style="color:${angleColor(deg)}">${deg}°</span>
    </div>
  `).join('');
}

function updateStatsTotalReps() {
  document.getElementById('stat-total-reps').textContent = state.totalReps + state.repCount;
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetReps() {
  fetch(`${API}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ exercise: state.currentExercise }),
  }).catch(() => {});
  resetRepsOnly();
  state.formScores = [];
  state.angleHistory = [];
  state.lastCoachingRep = -1;
  document.getElementById('stat-avg-score').textContent = '--';
  if (angleChart) { angleChart.data.labels = []; angleChart.data.datasets[0].data = []; angleChart.update(); }
}

function resetRepsOnly() {
  state.repCount = 0;
  document.getElementById('hud-reps').textContent = 0;
}

// ── AI Coaching ───────────────────────────────────────────────────────────────
async function triggerAutoCoaching(form, reps) {
  const advice = await fetchCoaching(form, reps.count);
  if (advice) updateCoachBubble(advice);
}

async function getAICoaching() {
  const advice = await fetchCoaching(null, state.repCount);
  if (advice) {
    updateCoachBubble(advice);
    showModal('🤖 AI Coach Feedback', advice.cue + '\n\n' + (advice.motivation || ''));
  }
}

async function fetchCoaching(form, repCount) {
  try {
    const res = await fetch(`${API}/coaching`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        exercise: state.currentExercise,
        form_score: form?.score || 80,
        corrections: form?.corrections || [],
        angles: form?.angles || {},
        rep_count: repCount,
        session_id: SESSION_ID,
        user_level: document.getElementById('fitness-level').value,
      }),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return {
      cue: "Backend not connected. Start app.py to enable AI coaching.",
      motivation: "You're doing great regardless!",
      voice_text: null,
    };
  }
}

function updateCoachBubble(advice) {
  document.getElementById('coach-cue').textContent = advice.cue || '';
  document.getElementById('coach-motivation').textContent = advice.motivation || '';
  if (advice.voice_text) speak(advice.voice_text);
}

// ── Workout plan ──────────────────────────────────────────────────────────────
async function generatePlan() {
  showModal('📅 Generating Workout Plan…', 'Asking Claude AI to create your personalised plan…');
  try {
    const res = await fetch(`${API}/workout-plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_profile: {
          fitness_level: document.getElementById('fitness-level').value,
          goals: 'Build strength and improve form',
          equipment: 'Bodyweight + dumbbells',
        },
        recent_sessions: [
          { exercise: state.currentExercise, sets: state.setCount, total_reps: state.totalReps + state.repCount }
        ],
      }),
    });
    if (!res.ok) throw new Error('Backend not running');
    const plan = await res.json();
    const text = formatPlan(plan);
    showModal('📅 Your Weekly Workout Plan', text);
  } catch {
    showModal('📅 Workout Plan', 'Could not connect to backend.\n\nStart app.py and set ANTHROPIC_API_KEY to generate personalised plans.');
  }
}

function formatPlan(plan) {
  if (!plan.days) return JSON.stringify(plan, null, 2);
  let text = `${plan.plan_name || 'Weekly Plan'}\n${plan.weekly_volume || ''}\n\n`;
  plan.days?.forEach(day => {
    text += `── ${day.day}: ${day.focus} ──\n`;
    day.exercises?.forEach(ex => {
      text += `  • ${ex.name}: ${ex.sets} sets × ${ex.reps} reps (rest ${ex.rest_seconds}s)\n`;
      if (ex.notes) text += `    ${ex.notes}\n`;
    });
    text += '\n';
  });
  if (plan.progression_notes) text += `Progression: ${plan.progression_notes}\n`;
  if (plan.nutrition_tip) text += `Nutrition tip: ${plan.nutrition_tip}\n`;
  return text;
}

// ── Voice ─────────────────────────────────────────────────────────────────────
function speak(text) {
  if (!document.getElementById('voice-toggle').checked) return;
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 1.05;
  utt.pitch = 1.0;
  utt.volume = 0.9;
  window.speechSynthesis.speak(utt);
}

// ── Modal ─────────────────────────────────────────────────────────────────────
function showModal(title, body) {
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-body').textContent = body;
  document.getElementById('modal-bg').classList.add('open');
}
function closeModal(e) { if (e.target.id === 'modal-bg') closeModalDirect(); }
function closeModalDirect() { document.getElementById('modal-bg').classList.remove('open'); }

// ── Helpers ───────────────────────────────────────────────────────────────────
function scoreClass(s) { return s >= 85 ? 'score-good' : s >= 60 ? 'score-warn' : 'score-bad'; }
function scoreColor(s) { return s >= 85 ? 'var(--green)' : s >= 60 ? 'var(--amber)' : 'var(--red)'; }
function angleColor(deg) {
  if (deg >= 160) return 'var(--green)';
  if (deg >= 100) return 'var(--amber)';
  return 'var(--red)';
}
function setStatus(type, label) {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  text.textContent = label;
  dot.style.background = type === 'active' ? 'var(--green)' : type === 'warn' ? 'var(--amber)' : 'var(--muted)';
}

// ── Init ──────────────────────────────────────────────────────────────────────
buildExerciseList();
buildSetTracker();
initChart();
selectExercise('squat');

// Check backend health
fetch(`http://localhost:5000/health`)
  .then(r => r.json())
  .then(d => {
    setStatus('active', d.coach_available ? '✓ AI Ready' : '⚠ No AI key');
    document.getElementById('coach-cue').textContent =
      d.coach_available
        ? 'AI Coach connected! Start your workout.'
        : 'Set ANTHROPIC_API_KEY to enable AI coaching.';
  })
  .catch(() => {
    setStatus('idle', 'Backend offline');
    document.getElementById('coach-cue').textContent =
      'Backend not running. Start backend/app.py first.';
  });

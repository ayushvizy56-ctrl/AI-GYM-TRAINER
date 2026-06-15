import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RepState:
    count: int = 0
    phase: str = "idle"
    last_phase: str = "idle"
    angle_history: deque = field(default_factory=lambda: deque(maxlen=30))
    smoothed_angle: float = 0.0


class RepCounter:
    THRESHOLDS = {
        "squat":          (100, 160),
        "bicep_curl":     (50,  150),
        "pushup":         (70,  160),
        "shoulder_press": (90,  165),
        "deadlift":       (120, 170),
        "lunge":          (85,  155),
    }

    EMA_ALPHA = 0.3

    def __init__(self):
        self._states: dict = {}

    def _get_state(self, exercise: str) -> RepState:
        if exercise not in self._states:
            self._states[exercise] = RepState()
        return self._states[exercise]

    def update(self, exercise: str, raw_angle: float) -> dict:
        state = self._get_state(exercise)
        low, high = self.THRESHOLDS.get(exercise, (90, 160))

        if state.smoothed_angle == 0.0:
            state.smoothed_angle = raw_angle
        else:
            state.smoothed_angle = (
                self.EMA_ALPHA * raw_angle +
                (1 - self.EMA_ALPHA) * state.smoothed_angle
            )
        angle = state.smoothed_angle
        state.angle_history.append(angle)

        prev = state.phase

        if angle <= low:
            new_phase = "bottom"
        elif angle >= high:
            new_phase = "top"
        else:
            new_phase = prev

        if prev == "bottom" and new_phase == "top":
            state.count += 1

        state.last_phase = prev
        state.phase = new_phase

        return {
            "count": state.count,
            "phase": new_phase,
            "smoothed_angle": round(angle, 1),
            "raw_angle": round(raw_angle, 1),
        }

    def reset(self, exercise: str = None):
        if exercise:
            self._states.pop(exercise, None)
        else:
            self._states.clear()

    def get_count(self, exercise: str) -> int:
        return self._get_state(exercise).count
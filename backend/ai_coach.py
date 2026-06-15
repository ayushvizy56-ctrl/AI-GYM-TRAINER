"""
ai_coach.py
------------
Uses the Anthropic Claude API to generate personalised coaching advice
based on the user's exercise form data, rep count, and history.

The coach speaks in short, direct, motivational sentences — like a real
personal trainer standing next to you.
"""

import anthropic
import json
import os
from typing import Optional


class AICoach:
    """
    Sends form data + user context to Claude and gets back coaching cues.

    Quick feedback (<500ms): rule-based from exercise_analyzer.py
    Deep coaching (on demand): Claude API call with full session context
    """

    SYSTEM_PROMPT = """You are an expert personal trainer and sports scientist.
Your job: give SHORT, DIRECT, ACTIONABLE feedback (2-3 sentences max).
- Always be encouraging but honest
- Use plain language, no jargon
- If form is good, motivate them to push harder
- If form is bad, tell them exactly ONE thing to fix (the most important one)
- Reference the specific angle data when relevant
- Keep it under 50 words
Format: JSON with keys "cue" (main feedback), "motivation" (1 motivating sentence), "priority_fix" (null or what to fix)
"""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY environment variable")
        self.client = anthropic.Anthropic(api_key=key)

    def get_coaching(
        self,
        exercise: str,
        form_score: int,
        corrections: list,
        angles: dict,
        rep_count: int,
        set_number: int = 1,
        user_level: str = "intermediate",
        session_reps_history: list = None,
    ) -> dict:
        """
        Generate personalised coaching advice.

        Returns:
            {
                "cue": "main feedback string",
                "motivation": "motivating sentence",
                "priority_fix": "what to fix" or null,
                "voice_text": "text for text-to-speech"
            }
        """
        session_summary = ""
        if session_reps_history:
            avg_score = sum(session_reps_history) / len(session_reps_history)
            session_summary = f"Session average form score: {avg_score:.0f}/100. "

        user_message = f"""
Exercise: {exercise}
Rep #{rep_count}, Set #{set_number}
Fitness level: {user_level}
Current form score: {form_score}/100
Joint angles: {json.dumps(angles)}
Corrections needed: {corrections if corrections else "none"}
{session_summary}

Give me coaching feedback for this rep.
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            text = response.content[0].text.strip()

            # Parse JSON response
            if text.startswith("{"):
                data = json.loads(text)
            else:
                # Fallback if model returns plain text
                data = {
                    "cue": text[:100],
                    "motivation": "Keep going!",
                    "priority_fix": corrections[0] if corrections else None,
                }

            # Build voice text (short and punchy)
            voice = data.get("cue", "")
            if data.get("priority_fix"):
                voice = f"{voice} Fix: {data['priority_fix']}."

            data["voice_text"] = voice
            return data

        except (json.JSONDecodeError, Exception) as e:
            # Graceful fallback — never block the workout
            return self._fallback_coaching(form_score, corrections, exercise, rep_count)

    def _fallback_coaching(self, score, corrections, exercise, rep_count) -> dict:
        """Rule-based fallback when API is unavailable."""
        if score >= 85:
            cue = f"Rep {rep_count} — great form! Keep that rhythm."
            fix = None
        elif score >= 60 and corrections:
            fix = corrections[0].replace("_", " ")
            cue = f"Watch your {fix} on the next rep."
        else:
            fix = corrections[0].replace("_", " ") if corrections else "form"
            cue = f"Slow down and focus on {fix}."

        return {
            "cue": cue,
            "motivation": "You've got this — stay focused!",
            "priority_fix": fix,
            "voice_text": cue,
        }

    def generate_workout_plan(
        self,
        user_profile: dict,
        recent_sessions: list,
    ) -> dict:
        """
        Generate a full weekly workout plan based on user profile + history.

        user_profile: {name, age, fitness_level, goals, equipment}
        recent_sessions: list of session summaries
        """
        prompt = f"""
Create a personalised 5-day workout plan for this user.

User profile: {json.dumps(user_profile, indent=2)}
Recent workout history: {json.dumps(recent_sessions[-5:], indent=2) if recent_sessions else "No history yet"}

Return JSON: {{
  "plan_name": "...",
  "weekly_volume": "...",
  "days": [
    {{
      "day": "Monday",
      "focus": "...",
      "exercises": [
        {{"name": "...", "sets": 3, "reps": "8-12", "rest_seconds": 60, "notes": "..."}}
      ]
    }}
  ],
  "progression_notes": "...",
  "nutrition_tip": "..."
}}
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])

    def analyze_session(self, session_data: dict) -> str:
        """Generate a post-workout analysis report."""
        prompt = f"""
Analyze this workout session and give a 3-paragraph post-workout report.
Include: what went well, what to improve, recovery recommendations.

Session data: {json.dumps(session_data, indent=2)}
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

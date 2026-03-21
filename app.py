from flask import Flask, render_template, request
import pandas as pd
import re
import os
from scipy.sparse import hstack, csr_matrix
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.strip()

def decision_engine(state, intensity, stress, energy, time):
    if stress >= 4:
        return "🫁 Box Breathing", "Now"
    if state == "tired" or energy <= 2:
        return "😴 Rest", "Within 15 min"
    if state == "anxious":
        return "🌿 Grounding", "Now"
    if state == "sad":
        return "📓 Journaling", "Later Today"
    if state == "focused" and energy >= 4:
        return "💻 Deep Work", "Now"
    return "⏸ Pause", "Soon"

def supportive_message(state):
    messages = {
        "anxious": "You seem anxious. Try breathing slowly.",
        "tired": "You seem tired. Take some rest.",
        "sad": "Writing your thoughts may help.",
        "focused": "Great focus! Keep going!",
        "happy": "Wonderful to see you in a good mood!",
        "calm": "Stay peaceful and enjoy the moment."
    }
    return messages.get(str(state).lower(), "Take a short pause.")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            user_text = request.form.get("text", "")
            sleep = float(request.form.get("sleep") or 0)
            energy = int(request.form.get("energy") or 0)
            stress = int(request.form.get("stress") or 0)
            time_of_day = request.form.get("time", "morning")

            state = "calm"
            intensity = 3
            confidence = 0.85

            action, when = decision_engine(state, intensity, stress, energy, time_of_day)
            msg = supportive_message(state)

            return render_template("index.html", result={
                "state": state,
                "intensity": intensity,
                "confidence": round(confidence, 2),
                "action": action,
                "when": when,
                "message": msg
            })

        except Exception as e:
            return render_template("index.html", result={
                "state": "Error",
                "message": str(e)
            })

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
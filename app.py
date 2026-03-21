from flask import Flask, render_template, request
import pickle
import pandas as pd
import re
import os
from scipy.sparse import hstack, csr_matrix
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

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
            emotion_model = load_model(os.path.join(BASE_DIR, "models/emotion_model.pkl"))
            intensity_model = load_model(os.path.join(BASE_DIR, "models/intensity_model.pkl"))
            vectorizer = load_model(os.path.join(BASE_DIR, "models/vectorizer.pkl"))
            train_meta_cols = load_model(os.path.join(BASE_DIR, "models/train_meta_cols.pkl"))

            if not all([emotion_model, intensity_model, vectorizer, train_meta_cols]):
                return render_template("index.html", result={
                    "state": "Error",
                    "message": "Model loading failed"
                })

            user_text = request.form.get("text", "")
            sleep = float(request.form.get("sleep") or 0)
            energy = int(request.form.get("energy") or 0)
            stress = int(request.form.get("stress") or 0)
            time_of_day = request.form.get("time", "morning")

            user_clean = clean_text(user_text)
            user_text_vec = vectorizer.transform([user_clean])

            user_meta = pd.DataFrame([{
                "duration_min": 10,
                "sleep_hours": sleep,
                "energy_level": energy,
                "stress_level": stress,
                "ambience_type": "unknown",
                "time_of_day": time_of_day,
                "previous_day_mood": "unknown",
                "face_emotion_hint": "unknown",
                "reflection_quality": "medium"
            }])

            user_meta = pd.get_dummies(user_meta)
            user_meta = user_meta.reindex(columns=train_meta_cols, fill_value=0)

            user_meta_sparse = csr_matrix(user_meta.astype(float).values)
            user_X = hstack([user_text_vec, user_meta_sparse])

            state = emotion_model.predict(user_X)[0]
            intensity = int(intensity_model.predict(user_X)[0])
            confidence = float(emotion_model.predict_proba(user_X).max())

            action, when = decision_engine(state, intensity, stress, energy, time_of_day)
            msg = supportive_message(state)

            if confidence < 0.4:
                msg = "🤔 I'm still learning, but " + msg

            return render_template("index.html", result={
                "state": state,
                "intensity": intensity,
                "confidence": round(confidence, 2),
                "action": action,
                "when": when,
                "message": msg
            })

        except Exception as e:
            print("ERROR:", e)
            return render_template("index.html", result={
                "state": "Error",
                "message": str(e)
            })

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
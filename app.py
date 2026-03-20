from flask import Flask, render_template, request
import pickle
import pandas as pd
import re
from scipy.sparse import hstack

app = Flask(__name__)

# ==========================================
# LOAD MODELS (FINAL CORRECT PATH)
# ==========================================
with open("models/emotion_model.pkl", "rb") as f:
    emotion_model = pickle.load(f)

with open("models/intensity_model.pkl", "rb") as f:
    intensity_model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/train_meta_cols.pkl", "rb") as f:
    train_meta_cols = pickle.load(f)

print("✅ Models loaded successfully")

# ==========================================
# TEXT CLEANING
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# ==========================================
# DECISION ENGINE
# ==========================================
def decision_engine(state, intensity, stress, energy, time):

    if stress >= 4 and intensity >= 3:
        return "🫁 Box Breathing", "Now"

    if state == "tired" and energy <= 2:
        return "😴 Rest", "Within 15 min"

    if state == "anxious":
        return "🌿 Grounding", "Now"

    if state == "sad":
        return "📓 Journaling", "Later Today"

    if state == "focused" and energy >= 4:
        return "💻 Deep Work", "Now"

    if time == "evening":
        return "📝 Light Planning", "Tonight"

    return "⏸ Pause", "Soon"

# ==========================================
# SUPPORT MESSAGE
# ==========================================
def supportive_message(state):
    if state == "anxious":
        return "You seem anxious. Try breathing slowly."
    if state == "tired":
        return "You seem tired. Take some rest."
    if state == "sad":
        return "Writing your thoughts may help."
    if state == "focused":
        return "Great focus! Keep going!"
    return "Take a short pause."

# ==========================================
# MAIN ROUTE
# ==========================================
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        try:
            # INPUT
            user_text = request.form.get("text", "")
            sleep = float(request.form.get("sleep", 0))
            energy = int(request.form.get("energy", 0))
            stress = int(request.form.get("stress", 0))
            time_of_day = request.form.get("time", "morning")

            # CLEAN TEXT
            user_clean = clean_text(user_text)

            # TEXT VECTOR
            user_text_vec = vectorizer.transform([user_clean])

            # METADATA
            user_meta_dict = {
                "duration_min": 10,
                "sleep_hours": sleep,
                "energy_level": energy,
                "stress_level": stress,
                "ambience_type": "unknown",
                "time_of_day": time_of_day,
                "previous_day_mood": "unknown",
                "face_emotion_hint": "unknown",
                "reflection_quality": "medium"
            }

            user_meta = pd.DataFrame([user_meta_dict])
            user_meta = pd.get_dummies(user_meta)
            user_meta = user_meta.reindex(columns=train_meta_cols, fill_value=0)
            user_meta = user_meta.astype(float)

            # COMBINE
            user_X = hstack([user_text_vec, user_meta])

            # PREDICT
            state = emotion_model.predict(user_X)[0]
            intensity = int(intensity_model.predict(user_X)[0])

            # CONFIDENCE
            probs = emotion_model.predict_proba(user_X)
            confidence = float(probs.max())

            # DECISION
            action, when = decision_engine(
                state, intensity, stress, energy, time_of_day
            )

            # MESSAGE
            msg = supportive_message(state)

            if confidence < 0.5:
                msg = "🤔 I'm not fully sure, but " + msg

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
                "intensity": "-",
                "confidence": 0,
                "action": "Fix input",
                "when": "",
                "message": str(e)
            })

    return render_template("index.html", result=None)

# ==========================================
# RUN APP
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)
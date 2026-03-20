from flask import Flask, render_template, request
import pickle
import pandas as pd
import re
from scipy.sparse import hstack

app = Flask(__name__)

# Load saved models (from your main file)
emotion_model = pickle.load(open("emotion_model.pkl", "rb"))
intensity_model = pickle.load(open("intensity_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
train_meta_cols = pickle.load(open("train_meta_cols.pkl", "rb"))

# SAME function from your code
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# SAME decision engine
def decision_engine(state, intensity, stress, energy, time):

    if stress >= 4 and intensity >= 3:
        return "box_breathing", "now"

    if state == "tired" and energy <= 2:
        return "rest", "within_15_min"

    if state == "anxious":
        return "grounding", "now"

    if state == "sad":
        return "journaling", "later_today"

    if state == "focused" and energy >= 4:
        return "deep_work", "now"

    if time == "evening":
        return "light_planning", "tonight"

    return "pause", "within_15_min"

# SAME message function
def supportive_message(state):
    if state == "anxious":
        return "You seem anxious. Try breathing."
    if state == "tired":
        return "You seem tired. Take rest."
    if state == "sad":
        return "Try writing your thoughts."
    if state == "focused":
        return "Great focus! Keep going."
    return "Take a pause."

# MAIN ROUTE
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        user_text = request.form["text"]
        sleep = float(request.form["sleep"])
        energy = int(request.form["energy"])
        stress = int(request.form["stress"])
        time_of_day = request.form["time"]

        # === SAME LOGIC AS YOUR CODE ===
        user_clean = clean_text(user_text)
        user_text_vec = vectorizer.transform([user_clean])

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

        user_X = hstack([user_text_vec, user_meta])

        # Prediction
        state = emotion_model.predict(user_X)[0]
        intensity = int(intensity_model.predict(user_X)[0])

        confidence = float(emotion_model.predict_proba(user_X).max())
            
       

        action, when = decision_engine(
            state, intensity, stress, energy, time_of_day
        )

        msg = supportive_message(state)

        return render_template("index.html", result={
            "state": state,
            "intensity": intensity,
            "confidence": round(confidence, 2),
            "action": action,
            "when": when,
            "message": msg
        })

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
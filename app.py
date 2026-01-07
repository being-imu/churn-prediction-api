from flask import Flask, render_template, request
import pickle
from preprocessing import preprocess_input

app = Flask(__name__)

# Load model once
with open("xgb_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

THRESHOLD = 0.3  # same as training

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None

    if request.method == "POST":
        try:
            X = preprocess_input(request.form)
            proba = model.predict_proba(X)[0][1]

            probability = round(proba, 3)
            result = "Likely to Churn ❌" if proba >= THRESHOLD else "Not Likely to Churn ✅"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)

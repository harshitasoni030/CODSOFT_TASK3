"""Flask app that serves the churn model.  Run:  python app.py"""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model" / "churn_model.pkl")
info = json.loads((BASE_DIR / "model" / "model_info.json").read_text())
FEATURES = info["features"]

app = Flask(__name__)


@app.route("/")
def home():
    best = info["results"][info["best_model"]]
    return render_template("index.html", features=FEATURES,
                           best_model=info["best_model"], metrics=best)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    row = {}
    for f in FEATURES:
        value = data.get(f["name"])
        if value is None or str(value).strip() == "":
            return jsonify({"error": f"Missing value for {f['name']}"}), 400
        if f["type"] == "category":
            row[f["name"]] = str(value)
        else:
            try:
                row[f["name"]] = float(value)
            except ValueError:
                return jsonify({"error": f"{f['name']} must be a number"}), 400

    df = pd.DataFrame([row], columns=[f["name"] for f in FEATURES])
    probability = float(model.predict_proba(df)[0][1])
    label = "CHURN" if probability >= 0.5 else "NOT CHURN"
    return jsonify({"prediction": label,
                    "churn_probability": round(probability * 100, 2)})


if __name__ == "__main__":
    app.run(debug=True)

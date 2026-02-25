from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model/hydration_model.joblib"))


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "dev-secret-key"

    @dataclass(frozen=True)
    class FormChoices:
        genders: tuple[str, ...] = ("Male", "Female", "Other")
        activity_levels: tuple[str, ...] = ("Low", "Moderate", "High")
        weather_conditions: tuple[str, ...] = (
            "Cold",
            "Mild",
            "Hot",
            "Humid",
            "Rainy",
        )

    choices = FormChoices()
    _model: Any | None = None

    def get_model() -> Any:
        nonlocal _model
        if _model is not None:
            return _model

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Run train_model.py first."
            )

        _model = joblib.load(MODEL_PATH)
        return _model

    def parse_float(name: str, value: str) -> float:
        if value is None or value.strip() == "":
            raise ValueError(f"{name} is required.")
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"{name} must be a number.")

    def parse_choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
        if value is None or value.strip() == "":
            raise ValueError(f"{name} is required.")
        if value not in allowed:
            raise ValueError(f"Invalid {name}.")
        return value

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            choices=choices,
            result=None,
            form_values={},
        )

    @app.post("/predict")
    def predict():
        try:
            age = parse_float("Age", request.form.get("age", ""))
            weight = parse_float("Weight", request.form.get("weight", ""))
            daily_water = parse_float(
                "Daily Water Intake",
                request.form.get("daily_water_intake", "")
            )

            gender = parse_choice(
                "Gender",
                request.form.get("gender", ""),
                choices.genders,
            )

            activity = parse_choice(
                "Physical Activity Level",
                request.form.get("physical_activity_level", ""),
                choices.activity_levels,
            )

            weather = parse_choice(
                "Weather",
                request.form.get("weather", ""),
                choices.weather_conditions,
            )

            if age <= 0 or weight <= 0 or daily_water < 0:
                raise ValueError("Please enter realistic positive values.")

            row = {
                "Age": age,
                "Weight (kg)": weight,
                "Gender": gender,
                "Daily Water Intake (liters)": daily_water,
                "Physical Activity Level": activity,
                "Weather": weather,
            }

            model = get_model()
            X = pd.DataFrame([row])
            prediction = model.predict(X)[0]

            return render_template(
                "index.html",
                choices=choices,
                result=str(prediction),
                form_values=request.form.to_dict(),
            )

        except FileNotFoundError as e:
            flash(str(e), "error")
            return redirect(url_for("index"))
        except Exception as e:
            flash(str(e), "error")
            return render_template(
                "index.html",
                choices=choices,
                result=None,
                form_values=request.form.to_dict(),
            )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
"""Thin wrapper around the joblib-persisted Random Forest so routes.py
doesn't need to know about sklearn internals or the exact feature order."""

import joblib
import pandas as pd
from flask import current_app

_cache = {}


def _load():
    if "bundle" not in _cache:
        bundle = joblib.load(current_app.config["MODEL_PATH"])
        _cache["bundle"] = bundle
    return _cache["bundle"]


def feature_columns():
    return _load()["feature_cols"]


def feature_importances():
    """Global feature importances from the trained Random Forest, as
    {feature_name: importance}, sorted descending. Same for every
    prediction — this describes what the model relies on overall, not
    this specific input."""
    bundle = _load()
    return bundle.get("feature_importances", {})


def explain(features: dict, top_n: int = 3):
    """Lightweight, honest explanation: for the TOP globally-important
    features, say whether this input's value is above or below what's
    typical (median) in the training set. This is NOT a SHAP-style
    per-instance attribution — it's a simpler 'what stands out about this
    input, on the dimensions the model relies on most' summary, appropriate
    for a small-sample prototype rather than claiming precise causal
    contribution."""
    bundle = _load()
    importances = bundle.get("feature_importances", {})
    medians = bundle.get("feature_medians", {})

    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    explanation = []
    for name, importance in ranked:
        value = features.get(name)
        median = medians.get(name)
        if value is None or median is None:
            continue
        direction = "higher than" if value > median else ("lower than" if value < median else "equal to")
        explanation.append({
            "feature": name,
            "value": round(float(value), 3),
            "typical_value": round(float(median), 3),
            "direction": direction,
            "importance": round(float(importance), 3),
        })
    return explanation


def predict(features: dict):
    """features: dict of {feature_name: value}. Missing keys are filled
    with 0.0 — acceptable here since the model's own median-imputation
    happened at training time on the historical dataset; callers should
    still supply hist_* fields whenever a module has prior cohorts."""
    bundle = _load()
    cols = bundle["feature_cols"]
    model = bundle["model"]

    row = {col: features.get(col, 0.0) for col in cols}
    X = pd.DataFrame([row], columns=cols)

    label = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])  # P(class == 1, "persistently difficult")

    return label, probability

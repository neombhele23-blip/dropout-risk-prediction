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


def has_history(features: dict) -> bool:
    """A module presentation can only be scored if it has at least one
    earlier presentation to draw historical features from."""
    n_prior = features.get("hist_n_prior_cohorts")
    return bool(n_prior) and features.get("hist_last_difficulty") is not None


def predict(features: dict):
    """features: dict of {feature_name: value}. Any feature the caller
    leaves out is filled with its training-set median (stored in the
    model bundle), NOT 0.0 -- a 0.0 for e.g. hist_avg_difficulty would be
    far outside anything seen in training and produce a confident but
    meaningless 'not difficult'. Callers must check has_history() first;
    modules with no prior presentations are not scored at all."""
    bundle = _load()
    cols = bundle["feature_cols"]
    model = bundle["model"]
    medians = bundle.get("feature_medians", {})

    row = {}
    for col in cols:
        value = features.get(col)
        row[col] = medians.get(col, 0.0) if value is None else value
    X = pd.DataFrame([row], columns=cols)

    label = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])  # P(class == 1, "persistently difficult")

    return label, probability

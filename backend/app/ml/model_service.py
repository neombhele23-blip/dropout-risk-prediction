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

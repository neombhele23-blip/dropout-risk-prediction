"""
RESK401 Group 13 — Dropout Risk Prediction (module-level)
Step 2: Train and evaluate the Random Forest classifier.

Only cohorts with at least one PRIOR presentation of the same module are
usable (a module's very first observed presentation has no history yet
to build features from) — this drops us from 22 to 15 rows. That is a
genuine, worth-stating-in-the-report limitation: 15 samples is small for
stratified 5-fold CV (~3 per fold). Results here should be read as a
proof-of-concept on OULAD as a structural proxy, not as a claim of
production-grade accuracy — consistent with the limitation your group
already flagged for the oral (OULAD isn't DUT's own data).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    "enrollment_count",
    "avg_studied_credits",
    "avg_prev_attempts",
    "avg_imd",
    "disability_rate",
    "distinction_rate",
    "module_presentation_length",
    "hist_avg_difficulty",
    "hist_last_difficulty",
    "hist_n_prior_cohorts",
    "hist_avg_enrollment",
]
TARGET_COL = "is_difficult"
N_FOLDS = 5
RANDOM_STATE = 42


def load_dataset():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "cohort_dataset.csv"))
    # Drop cohorts with no history yet — no historical features to predict from.
    usable = df.dropna(subset=["hist_avg_difficulty"]).reset_index(drop=True)
    return usable


def train_and_evaluate(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df[TARGET_COL]

    print(f"Usable module-cohort rows (with history): {len(df)}")
    print(f"Class balance: {y.value_counts().to_dict()}\n")

    # N_FOLDS capped by the smaller class's count, since stratified k-fold
    # needs at least k members of every class.
    n_folds = min(N_FOLDS, y.value_counts().min())
    if n_folds < N_FOLDS:
        print(f"NOTE: reduced to {n_folds}-fold CV — smallest class has only "
              f"{y.value_counts().min()} samples.\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,          # kept shallow — small-sample dataset, avoid overfitting
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    scoring = ["accuracy", "precision", "recall", "f1"]
    cv_results = cross_validate(clf, X, y, cv=skf, scoring=scoring)

    print("=== Stratified K-Fold Cross-Validation Results ===")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        print(f"{metric.capitalize():10s}: {scores.mean():.3f}  (per-fold: {np.round(scores, 3).tolist()})")

    # Fit final model on all usable data for deployment / feature importance.
    clf.fit(X, y)
    preds = clf.predict(X)

    print("\n=== Classification Report (fit on full usable set — training fit, not held-out) ===")
    print(classification_report(y, preds, target_names=["Not persistently difficult", "Persistently difficult"]))

    print("=== Confusion Matrix (training fit) ===")
    print(confusion_matrix(y, preds))

    print("\n=== Feature Importances ===")
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importances.to_string())

    model_path = os.path.join(os.path.dirname(__file__), "dropout_risk_model.joblib")
    joblib.dump({"model": clf, "feature_cols": FEATURE_COLS}, model_path)
    print(f"\nSaved trained model to {model_path}")

    return clf, cv_results, importances


if __name__ == "__main__":
    dataset = load_dataset()
    train_and_evaluate(dataset)

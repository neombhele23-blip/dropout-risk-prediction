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

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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


def baseline_predictions(df: pd.DataFrame):
    """Historical-persistence baseline: 'this cohort will be difficult if the
    LAST cohort of the same module was above the dataset's own difficulty
    threshold'. No model, no fitting — just carries the previous outcome
    forward. This is the honest bar a Random Forest needs to clear to be
    worth using: it shows whether the RF adds real predictive value beyond
    'assume next term looks like last term.'

    Threshold is re-derived as the median of THIS dataset's difficulty_rate
    (the 15-row usable set) rather than the original 22-row prepare_data.py
    threshold, since that value isn't persisted to cohort_dataset.csv. On a
    dataset this small the two are close in practice, but this is a known
    approximation — worth a one-line footnote in the report if cited.
    """
    threshold = df["difficulty_rate"].median()
    baseline_pred = (df["hist_last_difficulty"] > threshold).astype(int)
    return baseline_pred, threshold


def train_and_evaluate(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df[TARGET_COL]

    print(f"Usable module-cohort rows (with history): {len(df)}")
    print(f"Class balance: {y.value_counts().to_dict()}\n")

    # --- Simple baseline: does the Random Forest beat 'just assume next
    # term looks like last term'? ---
    baseline_pred, baseline_threshold = baseline_predictions(df)
    baseline_metrics = {
        "accuracy": accuracy_score(y, baseline_pred),
        "precision": precision_score(y, baseline_pred, zero_division=0),
        "recall": recall_score(y, baseline_pred, zero_division=0),
        "f1": f1_score(y, baseline_pred, zero_division=0),
    }
    print("=== Baseline: 'next cohort repeats last cohort's difficulty' ===")
    print(f"(threshold on difficulty_rate: {baseline_threshold:.3f})")
    for metric, value in baseline_metrics.items():
        print(f"{metric.capitalize():10s}: {value:.3f}")
    print()

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

    # Same-basis comparison: baseline vs RF, both scored on the full usable
    # set (neither is held-out CV) so the numbers are directly comparable.
    rf_full_fit_metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
    }
    print("\n=== Baseline vs Random Forest (both scored on full usable set) ===")
    print(f"{'Metric':10s} {'Baseline':>10s} {'RF':>10s}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        print(f"{metric.capitalize():10s} {baseline_metrics[metric]:>10.3f} {rf_full_fit_metrics[metric]:>10.3f}")

    # Feature medians — used both for imputing missing values at predict
    # time and for building a simple "why was this flagged" explanation
    # (is this input's value above/below what's typical in the training set).
    feature_medians = df[FEATURE_COLS].median().to_dict()

    model_path = os.path.join(os.path.dirname(__file__), "dropout_risk_model.joblib")
    joblib.dump({
        "model": clf,
        "feature_cols": FEATURE_COLS,
        "feature_importances": importances.to_dict(),
        "feature_medians": feature_medians,
    }, model_path)
    print(f"\nSaved trained model to {model_path}")

    # Also save baseline/RF comparison as JSON so the backend can surface it
    # (e.g. a "model info" panel) without re-running training.
    metrics_path = os.path.join(os.path.dirname(__file__), "baseline_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "baseline_description": "Predicts 'difficult' if the module's last presentation was above the difficulty threshold — no model, just carries the previous outcome forward.",
            "baseline_threshold": baseline_threshold,
            "baseline_metrics_full_fit": baseline_metrics,
            "random_forest_metrics_full_fit": rf_full_fit_metrics,
            "random_forest_cv_metrics": {m: float(cv_results[f"test_{m}"].mean()) for m in scoring},
            "n_folds_used": n_folds,
            "n_usable_rows": len(df),
            "note": "Full-fit metrics are NOT held-out — both baseline and RF are scored on the same training data for a fair side-by-side comparison. CV metrics are the held-out estimate for the RF alone.",
        }, f, indent=2)
    print(f"Saved baseline/RF comparison to {metrics_path}")

    return clf, cv_results, importances


if __name__ == "__main__":
    dataset = load_dataset()
    train_and_evaluate(dataset)
